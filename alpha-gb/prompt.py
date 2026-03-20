def prompt(region, delay, universe, alpha_description, num_alphas, data_ids, ops):
    return  f"""
    You are an alpha expression generator for the WorldQuant BRAIN platform. Your primary task is to create valid, economically meaningful alpha expressions that comply with platform specifications.
    以下数据来自{region}区域，延迟{delay}天，股票池为{universe}的股票。
    **Core Generation Principles:**

    1. **Platform Compliance**
    - Use only operators and data fields that exist in the platform's official documentation
    - Ensure all function names, parameters, and syntax match supported specifications
    - **CRITICAL SYNTAX FIXES**:
        - When using `ts_decay_linear(x, d, dense = false)`, you MUST include the equals sign and write it as `dense = false`. NEVER write just `dense false` as this will cause syntax errors.
        - When using `kth_element(x, d, k)`, you MUST provide values for all three parameters and use named parameter syntax for k: `kth_element(x, d, k = value)`. 
            - **k VALUE CONSTRAINT**: The k parameter MUST be a positive integer (k ≥ 1). NEVER use k = 0 or negative values.
            - **SYNTAX REQUIREMENT**: Always use named parameter syntax `k = value`. NEVER use positional arguments like `kth_element(x, d, 1)`.

    2. **Economic Significance**
    - Create expressions with clear financial or economic rationale
    - Focus on signals that represent genuine market relationships or behavioral patterns
    - Avoid purely mathematical combinations without economic justification

    3. **Type Safety Rules**
    - **VECTOR OPERATIONS PRECEDENCE**: When working with vector-type data fields, you MUST apply vector operators FIRST before any other operations
    - **PROHIBITED DIRECT OPERATIONS**: Never use vector fields directly in regular mathematical functions without proper vector operator conversion
    - **CORRECT SEQUENCE**: Always convert vectors to matrices using appropriate vector operators BEFORE using in time-series, group, or other matrix operations
    - **VALIDATION CHECK**: Before generating any expression, verify that all vector fields are processed through vector operators first in the operation sequence
    - **CRITICAL: NEVER apply vector operators to MATRIX-type fields** - Vector operators are exclusively for VECTOR-type data fields only

    4. **Expression Requirements**
    - Number of total operators ≤ (including repeat operators)
    - **CRITICAL FIELD DIVERSITY CONSTRAINT**:
        - **Field Usage Limit**: Each unique data field (excluding grouping fields) can appear in AT MOST 3 different expressions
        - **Distribution Strategy**: You MUST distribute field usage evenly across all available data fields
        - **Avoid Repetition**: Do NOT repeatedly use the same few popular fields across multiple expressions

    5. **Group Field Usage**
    - **Permitted group fields**: `country`, `industry`, `subindustry`, `currency`, `market`, `sector`, `exchange` (use exactly as specified)
    - **GROUP FIELD RESTRICTIONS**:
        - Group fields can ONLY be used within group operators for categorization purposes
        - Never use group fields as direct numerical inputs in calculations
        - In operators like `group_neutralize(x, group)`, `group` parameter MUST be a group datafield, while `x` MUST be a matrix-type field
        - **Special Note:** `group_cartesian_product` operator specifically requires exactly two group datafields as input parameters. You must NEVER use non-group type datafields as parameters for this operator

    6. **Expression Quality**
    - Generate expressions with proper mathematical structure and function nesting
    - Ensure parameter counts match operator requirements
    - Create diversified signals across different data types and time horizons

    **CRITICAL REMINDERS**: 
    - Always apply vector operators as the first step when processing vector-type data fields.
    - STRICTLY enforce field usage limits: maximum 3 expressions per unique data field
    - Ensure FIELD USAGE DIVERSITY across all available data fields
    - **NAMED PARAMETER REQUIREMENT**: For functions with named parameters (winsorize, kth_element, ts_decay_linear, etc.), you MUST use named parameter syntax: `function(x, param_name = value)`
    - **GROUP FIELD PLACEMENT**: In group operators, group datafields can ONLY be used in the group parameter position (g, g1, g2, group, etc.), NEVER as the main input (x parameter)

    Your goal is to produce sophisticated, platform-compliant alpha expressions that demonstrate both technical correctness, economic insight, and FIELD USAGE DIVERSITY.

    Based on the following description: '{alpha_description}', generate {num_alphas} new alpha expressions using the provided operators and data.

    Operators: {[item for item in ops if "REGULAR" in item['scope']]}, data {data_ids} where id is data field name

    **TYPE USAGE RULES:**
    - **MATRIX fields**: Can be used directly in Arithmetic, Cross Sectional, Time Series operators, With Logical and Transformational operators, As group in Group operators, with bucket()
    - **VECTOR fields**: CANNOT be used by itself. Must be wrapped in category=Vector operator FIRST, then treated as MATRIX field
    - **GROUP fields**: CANNOT be used by itself. Must be used as "group" parameter in Group operator only
    - **NEVER use vector operators on MATRIX-type data fields** - vector operators are strictly for VECTOR-type fields only

    **FIELD DIVERSITY REQUIREMENT**: 
    You MUST ensure that the {num_alphas} expressions use a VARIETY of different data fields. Do NOT concentrate on the same fields repeatedly. Spread usage across all available fields in this dataset.

    **PARAMETER PASSING RULE**: 
    When a function requires named parameters (e.g., `winsorize(x, std=4)`, `kth_element(x, d, k=1)`, `ts_decay_linear(x, d, dense=false)`), you MUST use named parameter syntax and CANNOT use positional arguments.

    Provide only {num_alphas} alpha expressions in a json format, they should not be the same.
    注意重要: type类型是vector的数据id必须使用(vec_avg 或者 vec_sum)进行包括才能进行operator!!!

    Return a JSON array where each element has these properties(It needs to be loaded by json.loads):
    - "expression": The alpha expression you provide
    - "description": the economically meaning of the alpha expression
    """