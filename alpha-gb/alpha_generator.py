import asyncio
import json
import os
import random
import re
import time
from typing import Any

import httpx
import pandas as pd
import wqb
from dotenv import load_dotenv

from check_alpha_status import validate_expression
from alpha_cache import check_if_alpha_already_simulated

logger = wqb.wqb_logger()
wqb.print(f"{logger.name = }")

DEFAULT_FIELD_LIMIT = 200
DEFAULT_PAGE_LIMIT = 50
DEFAULT_ALPHA_BATCH_SIZE = 15
DEFAULT_SIMULATION_MULTIPLE = 10
DEFAULT_SIMULATION_CONCURRENCY = 2
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.S)
JSON_CODE_BLOCK_PATTERN = re.compile(r"^```json\s*|\s*```$", flags=re.S)
TRUE_VALUES = {"1", "true", "yes", "on"}


class AlpagGenerator:
    def __init__(
        self,
        username: str,
        password: str,
        model_url: str,
        modelname: str,
        region: str,
        universe: str,
        delay: int | str,
        alpha_description: str = "",
        decay: int = 0,
        neutralization: str = "FAST",
        truncation: float = 0.08,
        pasteurization: str = "OFF",
        unitHandling: str = "ON",
        nanHandling: str = "OFF",
        maxTrade: str = "OFF",
        tags: list[str] | None = None,
        theme: bool = False,
        model_api_key: str | None = None,
    ) -> None:
        self.sess = wqb.WQBSession((username, password))
        self.region = region
        self.delay = delay
        self.universe = universe
        self.alpha_description = alpha_description
        self.model_url = model_url
        self.model_name = modelname
        self.model_apikey = model_api_key
        self.decay = decay
        self.neutralization=neutralization
        self.truncation=truncation
        self.pasteurization=pasteurization
        self.unitHandling=unitHandling
        self.nanHandling=nanHandling
        self.maxTrade=maxTrade
        self.tags = tags or []
        self.theme = theme

    def get_data_fields(
        self,
        data_ids: list[str] | tuple[str, ...],
        every_data_total: int = DEFAULT_FIELD_LIMIT,
        page_limit: int = DEFAULT_PAGE_LIMIT,
    ) -> list[dict[str, Any]]:
        fields: list[dict[str, Any]] = []
        for dataset_id in data_ids:
            dataset_fields = self._get_dataset_fields(
                dataset_id=dataset_id,
                every_data_total=every_data_total,
                page_limit=page_limit,
            )
            fields.extend(dataset_fields)
        return fields

    def _get_dataset_fields(
        self,
        dataset_id: str,
        every_data_total: int,
        page_limit: int,
    ) -> list[dict[str, Any]]:
        total_fields = self.sess.search_fields_limited(
            region=self.region,
            delay=self.delay,
            universe=self.universe,
            dataset_id=dataset_id,
        ).json()["count"]
        fetch_count = min(total_fields, every_data_total)

        logger.info(
            f"数据集 {dataset_id} 共 {total_fields} 个字段，本次最多读取 {fetch_count} 个"
        )

        dataset_fields: list[dict[str, Any]] = []
        for offset in range(0, fetch_count, page_limit):
            response = self.sess.search_fields_limited(
                region=self.region,
                universe=self.universe,
                delay=self.delay,
                dataset_id=dataset_id,
                limit=page_limit,
                offset=offset,
                theme=self.theme,
            )
            dataset_fields.extend(response.json()["results"])
        return dataset_fields

    def get_operators(self) -> list[dict[str, Any]]:
        return self.sess.search_operators().json()

    @staticmethod
    def handle_response(response_data: dict[str, Any]) -> list[str]:
        choices = response_data.get("choice") or response_data.get("choices")
        if not choices:
            logger.error("模型返回数据缺少 choice/choices 字段，无法提取 alpha ideas")
            return []

        content = choices[0]["message"]["content"]
        content = THINK_BLOCK_PATTERN.sub("", content)
        content = JSON_CODE_BLOCK_PATTERN.sub("", content.strip())

        try:
            alpha_ideas_json = json.loads(content)
        except json.JSONDecodeError:
            logger.exception("模型返回内容解析失败")
            return []

        return [
            item["expression"]
            for item in alpha_ideas_json
            if isinstance(item, dict) and item.get("expression")
        ]

    def generator_idear(
        self,
        fields: list[dict[str, Any]],
        ops: list[dict[str, Any]],
        num_alphs: int,
    ) -> list[str]:
        prompt = self._build_prompt(fields=fields, ops=ops, num_alphs=num_alphs)
        headers = {
            "Authorization": f"Bearer {self.model_apikey}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40
        }

        logger.info("正在获取 alpha ideas")
        response = httpx.post(url=self.model_url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        return self.handle_response(response.json())

    def _build_prompt(
        self,
        fields: list[dict[str, Any]],
        ops: list[dict[str, Any]],
        num_alphs: int,
    ) -> str:
        import prompt as prompt_module

        field_context = [
            {
                "id": field.get("id"),
                "type": field.get("type"),
                "description": field.get("description"),
            }
            for field in fields
            if isinstance(field, dict)
        ]
        return prompt_module.prompt(
            region=self.region,
            delay=self.delay,
            universe=self.universe,
            alpha_description=self.alpha_description,
            num_alphas=num_alphs,
            data_ids=field_context,
            ops=ops,
        )

    def alpha_back(self, alphas: list[str]) -> int | None:
        alpha_objs = [self._build_alpha_payload(alpha) for alpha in alphas]
        alpha_objs = self._distinct_alphs(alpha_objs)
        if len(alpha_objs) == 0:
            return

        def on_success(context: dict[str, object]) -> None:
            session = context.get(self)
            if not isinstance(session, wqb.WQBSession):
                return

            children = context["resp"].json().get("children", [])
            for child in children:
                response = session.get(
                    f"https://api.worldquantbrain.com/simulations/{child}"
                )
                alpha_id = response.json()["alpha"]
                session.patch_properties(
                    alpha_id=alpha_id,
                    tags=self.tags,
                    log=logger,
                )

        multi_alphas = wqb.to_multi_alphas(
            alphas=alpha_objs,
            multiple=DEFAULT_SIMULATION_MULTIPLE,
        )
        responses = asyncio.run(
            self.sess.concurrent_simulate(
                multi_alphas,
                concurrency=DEFAULT_SIMULATION_CONCURRENCY,
                on_success=on_success,
                return_exceptions=True,
            )
        )

        return sum(
            1
            for response in responses
            if isinstance(response, httpx.Response) and 200 <= response.status_code < 300
        )

    def _build_alpha_payload(self, alpha: str) -> dict[str, Any]:
        return {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": self.region,
                "universe": self.universe,
                "delay": self.delay,
                "decay": self.decay,
                "neutralization": self.neutralization,
                "truncation": self.truncation,
                "pasteurization": self.pasteurization,
                "unitHandling": self.unitHandling,
                "nanHandling": self.nanHandling,
                "maxTrade": self.maxTrade,
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": alpha,
        }
    
    def _distinct_alphs(self, alpha_list: list) -> list:
        dis_alphas = []
        for alpha in alpha_list:
            if check_if_alpha_already_simulated(alpha):
                continue
            dis_alphas.append(alpha)    
        return dis_alphas
        

def parse_data_ids(value: str | None) -> list[str]:
    if not value:
        return []
    value = value.strip()
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            pass
        else:
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def get_env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def get_env_int(name: str, default: int) -> int:
    value = get_env_str(name)
    return int(value) if value is not None else default


def get_env_float(name: str, default: float) -> float:
    value = get_env_str(name)
    return float(value) if value is not None else default


def get_env_bool(name: str, default: bool = False) -> bool:
    value = get_env_str(name)
    if value is None:
        return default
    return value.lower() in TRUE_VALUES


def get_env_list(name: str) -> list[str]:
    value = get_env_str(name)
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def export_fields_to_csv(
    fields: list[dict[str, Any]],
    region: str,
    universe: str,
    delay: int | str,
) -> str:
    if not fields:
        raise ValueError("未获取到任何字段数据，请检查 DATA_IDS、账号权限或查询条件")

    rows = [
        {
            "id": field.get("id"),
            "type": field.get("type"),
            "description": field.get("description"),
        }
        for field in fields
        if isinstance(field, dict)
    ]
    dataframe = pd.DataFrame(rows)

    required_columns = ["id", "type", "description"]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        sample_keys = list(fields[0].keys()) if isinstance(fields[0], dict) else type(fields[0]).__name__
        raise ValueError(
            f"字段结构异常，缺少列: {missing_columns}，首条字段可用键: {sample_keys}"
        )

    dataframe = dataframe[required_columns].dropna(subset=["id"])
    if dataframe.empty:
        sample_keys = list(fields[0].keys()) if isinstance(fields[0], dict) else type(fields[0]).__name__
        raise ValueError(
            f"字段数据为空或缺少 id/type/description，有效键示例: {sample_keys}"
        )

    csv_name = f"{region}_{universe}_{delay}_{random.randint(0, 50)}.csv"
    dataframe.to_csv(csv_name, index=False, encoding="utf-8")
    logger.info(f"字段数据已保存，共 {len(dataframe)} 条")
    return csv_name


def collect_valid_ideas(ideas: list[str], csv_path: str) -> list[str]:
    valid_ideas: list[str] = []
    for idea in ideas:
        is_valid, _ = validate_expression(expression=idea, csv_path=csv_path)
        if is_valid:
            valid_ideas.append(idea)
    return valid_ideas


def build_generator_from_env() -> tuple[AlpagGenerator, list[str]]:
    load_dotenv()

    generator = AlpagGenerator(
        username=get_env_str("USERNAME"),
        password=get_env_str("PASSWARD"),
        model_url=get_env_str("MODEL_URL"),
        modelname=get_env_str("MODEL_NAME"),
        model_api_key=get_env_str("MODEL_APIKEY"),
        universe=get_env_str("UNIVERSE"),
        delay=get_env_int("DELAY", 1),
        region=get_env_str("REGION"),
        alpha_description=get_env_str(
            "ALPHA_DESCRIPTION",
            "Generate diversified, platform-compliant alpha expressions with clear economic rationale.",
        ),
        decay=get_env_int("DECAY", 0),
        neutralization=get_env_str("NEUTRALIZATION", "FAST"),
        truncation=get_env_float("TRUNCATION", 0.08),
        pasteurization=get_env_str("PASTEURIZATION", "OFF"),
        unitHandling=get_env_str("UNIT_HANDLING", "VERIFY"),
        nanHandling=get_env_str("NAN_HANDLING", "OFF"),
        maxTrade=get_env_str("MAX_TRADE", "OFF"),
        tags=get_env_list("TAGS"),
        theme=get_env_bool("THEME", False),
    )
    data_ids = parse_data_ids(os.getenv("DATA_IDS"))
    return generator, data_ids


def main() -> int:
    alpha_generator, data_ids = build_generator_from_env()
    batch = 1
    total_alphas = 0

    fields = alpha_generator.get_data_fields(data_ids=data_ids)
    csv_name = export_fields_to_csv(
        fields=fields,
        region=alpha_generator.region,
        universe=alpha_generator.universe,
        delay=alpha_generator.delay,
    )
    ops = alpha_generator.get_operators()

    while True:
        try:
            logger.info(f"batch {batch} 开始生成 alpha ideas")
            ideas = alpha_generator.generator_idear(
                fields=fields,
                ops=ops,
                num_alphs=DEFAULT_ALPHA_BATCH_SIZE,
            )
            valid_ideas = collect_valid_ideas(ideas=ideas, csv_path=csv_name)

            logger.info(
                "累计生成 alpha %s，本次生成 %s 个，本次有效 %s 个",
                total_alphas,
                len(ideas),
                len(valid_ideas),
            )
            if not valid_ideas:
                logger.info("本次没有有效 alpha idea，跳过回测")
                continue

            alpha_generator.alpha_back(valid_ideas)
            total_alphas += len(valid_ideas)
            batch += 1
        except KeyboardInterrupt:
            logger.info("generator stop")
            break
        except Exception:
            logger.exception("batch error, sleep 2min")
            time.sleep(120)

    return 0


def mian() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
