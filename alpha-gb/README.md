# alpha-gb

一个简单的 alpha 生成与回测脚本项目。

## 使用方式

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 修改 `.env` 中的配置，至少需要按实际情况填写：

- `USERNAME`
- `PASSWARD`
- `MODEL_URL`
- `MODEL_NAME`
- `MODEL_APIKEY`

其他参数如 `REGION`、`UNIVERSE`、`DATA_IDS`、`ALPHA_DESCRIPTION` 也可以按需要调整。

3. 启动脚本：

```bash
python alpha_generator.py
```

## 说明

- 项目会从 `.env` 读取配置。
- 运行过程中会生成字段 CSV、日志以及本地缓存文件。
