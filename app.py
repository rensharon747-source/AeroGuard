import os
import re
import io
import math
import asyncio
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import edge_tts
except ImportError:
    edge_tts = None


warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Airport Operations Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_EXCEL = "Copy of Track 3 Data.xlsx"

BACKGROUND_URL = (
    "https://images.unsplash.com/photo-1436491865332-7a61a109cc05"
    "?auto=format&fit=crop&w=2200&q=85"
)


CSS = """
<style>
:root {
    --navy: #07182b;
    --panel: rgba(9, 27, 46, 0.88);
    --panel-light: rgba(18, 43, 66, 0.87);
    --cyan: #39c5f6;
    --gold: #ffbd45;
    --white: #f5f8ff;
    --muted: #c4d2e3;
}

.stApp {
    background:
        linear-gradient(
            115deg,
            rgba(4, 16, 30, 0.94),
            rgba(7, 30, 51, 0.80),
            rgba(5, 18, 34, 0.92)
        ),
        url("BACKGROUND_PLACEHOLDER");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: var(--white);
}

[data-testid="stHeader"] {
    background: rgba(5, 17, 30, 0.40);
}

[data-testid="stSidebar"] {
    background: rgba(5, 18, 33, 0.96);
    border-right: 1px solid rgba(57, 197, 246, 0.30);
}

[data-testid="stSidebar"] * {
    color: #f3f7ff;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
}

h1, h2, h3, h4 {
    color: #f5f8ff !important;
    letter-spacing: -0.3px;
}

p, li, label, .stMarkdown, .stCaption {
    color: #e3ebf7;
}

.hero {
    padding: 28px 30px 24px 30px;
    border: 1px solid rgba(92, 191, 246, 0.40);
    border-radius: 22px;
    background: linear-gradient(
        130deg,
        rgba(8, 27, 47, 0.94),
        rgba(17, 51, 76, 0.78)
    );
    box-shadow: 0 16px 44px rgba(0, 0, 0, 0.24);
    margin-bottom: 22px;
}

.hero-title {
    color: #f7faff;
    font-size: clamp(28px, 3.2vw, 44px);
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 9px;
}

.hero-subtitle {
    color: #bfeaff;
    font-size: 16px;
    line-height: 1.6;
}

.glass {
    background: rgba(7, 25, 43, 0.89);
    border: 1px solid rgba(125, 194, 235, 0.29);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.17);
    margin-bottom: 16px;
}

.metric-card {
    background: rgba(9, 29, 49, 0.91);
    border: 1px solid rgba(57, 197, 246, 0.28);
    border-radius: 15px;
    padding: 17px 18px;
    min-height: 108px;
}

.metric-label {
    font-size: 13px;
    color: #c4d2e3;
    margin-bottom: 9px;
}

.metric-value {
    font-size: 29px;
    font-weight: 800;
    color: #f6fbff;
}

.metric-note {
    font-size: 12px;
    color: #b8d2e8;
    margin-top: 5px;
}

.section-label {
    color: #7bdcff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.answer-box {
    padding: 22px;
    border: 1px solid rgba(57, 197, 246, 0.36);
    border-radius: 18px;
    background: rgba(5, 21, 38, 0.96);
    line-height: 1.85;
    color: #f4f8ff;
    white-space: pre-wrap;
}

.notice {
    padding: 13px 16px;
    border-radius: 12px;
    background: rgba(255, 189, 69, 0.10);
    border: 1px solid rgba(255, 189, 69, 0.42);
    color: #ffe3ad;
    margin: 10px 0;
}

div.stButton > button {
    width: 100%;
    min-height: 47px;
    border-radius: 12px;
    border: 1px solid rgba(95, 213, 255, 0.48);
    background: linear-gradient(105deg, #087ed0, #08a9bc);
    color: white;
    font-weight: 700;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    border-color: #c0f1ff;
    color: white;
    background: linear-gradient(105deg, #0794e9, #0bbfc8);
}

div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    min-height: 52px;
    border-radius: 12px;
    border: 1px solid rgba(95, 213, 255, 0.48);
    background: linear-gradient(105deg, #087ed0, #08a9bc);
    color: white;
    font-weight: 800;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    color: white;
    border-color: #c0f1ff;
}

textarea, input {
    color: #f5f8ff !important;
}

[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="select"] > div {
    background: rgba(8, 23, 40, 0.96) !important;
    border-color: rgba(170, 207, 236, 0.52) !important;
}

[data-baseweb="select"] span {
    color: #f5f8ff !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(125, 194, 235, 0.28);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: rgba(125, 194, 235, 0.25);
}

footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    .hero {
        padding: 20px;
    }

    .metric-value {
        font-size: 24px;
    }
}
</style>
""".replace("BACKGROUND_PLACEHOLDER", BACKGROUND_URL)

st.markdown(CSS, unsafe_allow_html=True)


if "assistant_answer" not in st.session_state:
    st.session_state.assistant_answer = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "natural_speech" not in st.session_state:
    st.session_state.natural_speech = ""

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "model_result" not in st.session_state:
    st.session_state.model_result = None


def normalize_name(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def find_column(df, candidates):
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    for column in df.columns:
        name = str(column).strip().lower()
        for candidate in candidates:
            if candidate.strip().lower() in name:
                return column

    return None


def numeric_column(df, candidates, default=np.nan):
    column = find_column(df, candidates)

    if column is None:
        return pd.Series(default, index=df.index, dtype=float)

    values = (
        df[column]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    result = pd.to_numeric(values, errors="coerce")

    if result.notna().any():
        median = result.dropna().median()

        if median > 1 and median <= 100:
            result = result / 100.0

    return result


def text_column(df, candidates, default="Unknown"):
    column = find_column(df, candidates)

    if column is None:
        return pd.Series(default, index=df.index, dtype=object)

    return df[column].fillna(default).astype(str).str.strip()


def read_excel_sheets(source):
    workbook = pd.ExcelFile(source)

    sheets = {}

    for sheet_name in workbook.sheet_names:
        sheets[sheet_name] = pd.read_excel(
            workbook,
            sheet_name=sheet_name
        )

    return sheets


def load_track3_data(source):
    sheets = read_excel_sheets(source)

    def get_sheet(prefix):
        for name, frame in sheets.items():
            if name.startswith(prefix):
                return frame.copy()
        return pd.DataFrame()

    baseline = get_sheet("01_机场月度运行基线")
    scale = get_sheet("02_机场年度规模")
    weather = get_sheet("03_机场天气月度")
    events = get_sheet("04_模拟事件")
    rules = get_sheet("05_运营规则")
    outputs = get_sheet("06_智能助手输出")
    feedback = get_sheet("07_人工反馈")
    tests = get_sheet("08_测试问题")
    quality = get_sheet("11_数据质量问题")

    if baseline.empty:
        raise ValueError(
            "The workbook does not contain the expected monthly airport baseline sheet."
        )

    year_col = find_column(baseline, ["年份", "Year"])
    month_col = find_column(baseline, ["月份", "Month"])
    airport_col = find_column(baseline, ["机场名称", "机场", "Airport"])

    if airport_col is None:
        raise ValueError("The airport name column could not be found.")

    baseline["Airport"] = baseline[airport_col].astype(str).str.strip()

    if year_col is not None:
        baseline["Year"] = pd.to_numeric(
            baseline[year_col],
            errors="coerce"
        )
    else:
        baseline["Year"] = np.nan

    if month_col is not None:
        baseline["Month"] = pd.to_numeric(
            baseline[month_col],
            errors="coerce"
        )
    else:
        baseline["Month"] = np.nan

    baseline["ScheduleRate"] = numeric_column(
        baseline,
        [
            "按计划完成航班的比例",
            "按计划完成航班比例",
            "计划完成比例",
            "ScheduleRate",
        ]
    )

    baseline["ReleaseRate"] = numeric_column(
        baseline,
        [
            "机场放行正常比例",
            "放行正常比例",
            "ReleaseRate",
        ]
    )

    baseline["WOHI"] = (
        baseline["ScheduleRate"] * 0.6
        + baseline["ReleaseRate"] * 0.4
    ) * 100

    baseline["WOHI"] = baseline["WOHI"].clip(0, 100)

    baseline["AirportType"] = text_column(
        baseline,
        ["机场分类", "机场类别", "AirportType"],
        "Unknown"
    )

    baseline["AirportType"] = baseline["AirportType"].replace(
        {"nan": "Unknown", "None": "Unknown"}
    )

    baseline["Date"] = pd.to_datetime(
        {
            "year": baseline["Year"].fillna(2025).astype(int),
            "month": baseline["Month"].fillna(1).astype(int).clip(1, 12),
            "day": 1,
        },
        errors="coerce"
    )

    baseline = baseline.dropna(
        subset=["Date", "WOHI"]
    ).copy()

    baseline = baseline.sort_values(
        ["Airport", "Date"]
    ).reset_index(drop=True)

    if not weather.empty:
        weather_airport_col = find_column(
            weather,
            ["机场名称", "机场", "Airport"]
        )

        weather_year_col = find_column(weather, ["年份", "Year"])
        weather_month_col = find_column(weather, ["月份", "Month"])

        if weather_airport_col is not None:
            weather["Airport"] = (
                weather[weather_airport_col].astype(str).str.strip()
            )
        else:
            weather["Airport"] = ""

        if weather_year_col is not None:
            weather["Year"] = pd.to_numeric(
                weather[weather_year_col],
                errors="coerce"
            )
        else:
            weather["Year"] = np.nan

        if weather_month_col is not None:
            weather["Month"] = pd.to_numeric(
                weather[weather_month_col],
                errors="coerce"
            )
        else:
            weather["Month"] = np.nan

        weather["Date"] = pd.to_datetime(
            {
                "year": weather["Year"].fillna(2025).astype(int),
                "month": weather["Month"].fillna(1).astype(int).clip(1, 12),
                "day": 1,
            },
            errors="coerce"
        )

        weather["Rainfall"] = numeric_column(
            weather,
            ["降水量", "降水量（毫米）", "Rainfall"]
        )

        weather["WindSpeed"] = numeric_column(
            weather,
            ["最大风速", "最大风速（米/秒）", "WindSpeed"]
        )

        weather["BadWeatherDays"] = numeric_column(
            weather,
            ["明显不利天气天数", "不利天气天数", "BadWeatherDays"]
        )

        weather["Temperature"] = numeric_column(
            weather,
            ["平均气温", "平均气温（摄氏度）", "Temperature"]
        )

        weather["WeatherRisk"] = calculate_weather_risk(weather)

        weather = weather.dropna(
            subset=["Date"]
        ).copy()

        baseline = baseline.merge(
            weather[
                [
                    "Airport",
                    "Date",
                    "Rainfall",
                    "WindSpeed",
                    "BadWeatherDays",
                    "Temperature",
                    "WeatherRisk",
                ]
            ],
            on=["Airport", "Date"],
            how="left"
        )

    for column in [
        "Rainfall",
        "WindSpeed",
        "BadWeatherDays",
        "Temperature",
        "WeatherRisk",
    ]:
        if column not in baseline.columns:
            baseline[column] = np.nan

    baseline["WeatherRisk"] = baseline["WeatherRisk"].fillna(
        baseline.groupby("Airport")["WeatherRisk"].transform("median")
    )

    baseline["WeatherRisk"] = baseline["WeatherRisk"].fillna(0)

    baseline["WOHI_Mean"] = baseline.groupby("Airport")["WOHI"].transform(
        lambda series: series.rolling(
            window=6,
            min_periods=1
        ).mean()
    )

    baseline["WOHI_Std"] = baseline.groupby("Airport")["WOHI"].transform(
        lambda series: series.rolling(
            window=6,
            min_periods=2
        ).std()
    )

    baseline["WOHI_Std"] = baseline["WOHI_Std"].replace(
        0, np.nan
    )

    baseline["AnomalyScore"] = (
        baseline["WOHI"] - baseline["WOHI_Mean"]
    ) / baseline["WOHI_Std"]

    baseline["AnomalyScore"] = baseline["AnomalyScore"].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    if not scale.empty:
        scale_airport_col = find_column(
            scale,
            ["机场名称", "机场", "Airport"]
        )

        if scale_airport_col is not None:
            scale["Airport"] = (
                scale[scale_airport_col].astype(str).str.strip()
            )

            passenger_col = find_column(
                scale,
                ["旅客吞吐量", "旅客吞吐量（万人次）", "Passengers"]
            )

            importance_col = find_column(
                scale,
                ["航空公司业务重要程度", "业务重要程度", "Importance"]
            )

            if passenger_col is not None:
                scale["Passengers"] = pd.to_numeric(
                    scale[passenger_col].astype(str).str.replace(",", "", regex=False),
                    errors="coerce"
                )
            else:
                scale["Passengers"] = np.nan

            if importance_col is not None:
                scale["Importance"] = scale[importance_col].astype(str)
            else:
                scale["Importance"] = "Unknown"

            scale = scale.drop_duplicates(
                subset=["Airport"],
                keep="last"
            )

            baseline = baseline.merge(
                scale[
                    ["Airport", "Passengers", "Importance"]
                ],
                on="Airport",
                how="left"
            )

    if "Passengers" not in baseline.columns:
        baseline["Passengers"] = np.nan

    if "Importance" not in baseline.columns:
        baseline["Importance"] = "Unknown"

    return {
        "sheets": sheets,
        "baseline": baseline,
        "scale": scale,
        "weather": weather if "weather" in locals() else pd.DataFrame(),
        "events": events,
        "rules": rules,
        "outputs": outputs,
        "feedback": feedback,
        "tests": tests,
        "quality": quality,
    }


def calculate_weather_risk(weather):
    rain = numeric_column(
        weather,
        ["降水量", "降水量（毫米）", "Rainfall"]
    )

    wind = numeric_column(
        weather,
        ["最大风速", "最大风速（米/秒）", "WindSpeed"]
    )

    bad_days = numeric_column(
        weather,
        ["明显不利天气天数", "不利天气天数", "BadWeatherDays"]
    )

    rain_score = rain.fillna(rain.median()).fillna(0).clip(0, 200) / 200
    wind_score = wind.fillna(wind.median()).fillna(0).clip(0, 35) / 35
    days_score = bad_days.fillna(bad_days.median()).fillna(0).clip(0, 31) / 31

    score = (
        rain_score * 0.35
        + wind_score * 0.40
        + days_score * 0.25
    ) * 100

    return score.clip(0, 100)


def load_uploaded_or_local(uploaded_file, local_path):
    if uploaded_file is not None:
        return load_track3_data(uploaded_file)

    if os.path.exists(local_path):
        return load_track3_data(local_path)

    return None


def get_latest_airport_data(data, airport):
    frame = data["baseline"]

    selected = frame[
        frame["Airport"].astype(str) == str(airport)
    ].sort_values("Date")

    return selected.copy()


def make_fallback_answer(
    question,
    airport,
    role,
    language,
    current_data,
    rules
):
    question_lower = question.lower()

    latest = None

    if current_data is not None and not current_data.empty:
        latest = current_data.sort_values("Date").iloc[-1]

    if language == "zh":
        wohi = (
            f"{latest['WOHI']:.1f}"
            if latest is not None
            else "暂无"
        )

        weather_risk = (
            f"{latest['WeatherRisk']:.1f}"
            if latest is not None
            else "暂无"
        )

        current_status = (
            f"现有月度运行健康指数为 {wohi}，"
            f"历史天气风险指标为 {weather_risk}。"
            if latest is not None
            else "当前没有足够的机场月度记录来计算运行指标。"
        )

        if role == "旅客":
            return (
                f"关于{airport}，建议首先确认航空公司发布的最新航班状态。"
                f"{current_status}"
                "这些数据是赛事提供的模拟数据，不能用于确认真实航班状态。"
                "目前不能仅凭天气或历史指标断定航班一定延误或取消。"
                "请以航空公司和机场的正式通知为准。"
            )

        if any(
            term in question_lower
            for term in ["priority", "priorities", "优先", "首先", "下一步"]
        ):
            return (
                f"{airport}当前建议按三个顺序处理。"
                "第一，核实最新天气预警、有效时段以及跑道和机位的实际条件。"
                "第二，与航空公司确认航班计划、飞机后续执行、机组和地面保障状态。"
                "第三，根据核实结果检查适用的运营规则，"
                "必要时将情况升级给负责的运行管理人员。"
                f"{current_status}"
                "目前缺少逐航班实时数据，因此不能认定某个航班必然受到影响。"
                "所有实际运行调整都应由授权人员评估和审批。"
            )

        return (
            f"关于{airport}，建议先核实问题涉及的最新现场信息。"
            f"{current_status}"
            "随后应检查相关运营规则、信息缺口和可能的风险，"
            "并由责任人员决定是否需要升级处理。"
            "这里的模型和指标仅用于赛事模拟与辅助决策，"
            "不能替代实时运行数据或人工审批。"
        )

    wohi = (
        f"{latest['WOHI']:.1f}"
        if latest is not None
        else "unavailable"
    )

    weather_risk = (
        f"{latest['WeatherRisk']:.1f}"
        if latest is not None
        else "unavailable"
    )

    current_status = (
        f"The latest monthly operational health index is {wohi}, "
        f"and the historical weather risk indicator is {weather_risk}. "
        if latest is not None
        else "There is not enough monthly airport data to calculate the current indicators. "
    )

    if role == "Passengers":
        return (
            f"For {airport}, the first priority is to check the latest "
            "official flight status from the airline. "
            + current_status
            + "These figures come from simulated competition data and do not "
            "confirm the status of any real flight. The available information "
            "does not prove that a flight will be delayed or cancelled. "
            "Please follow official airline and airport updates."
        )

    if any(
        term in question_lower
        for term in ["priority", "priorities", "first", "next step"]
    ):
        return (
            f"For {airport}, I recommend three priorities. First, verify the "
            "latest weather warning, its effective period, and the actual "
            "runway and stand conditions. Second, confirm flight schedules, "
            "aircraft rotations, crew availability, and ground handling with "
            "the relevant airline and operations teams. Third, check the "
            "applicable operational rules and escalate the situation if the "
            "verified facts meet the relevant criteria. "
            + current_status
            + "We do not have live flight-level data, so we cannot conclude "
            "that a particular flight will be disrupted. Any actual operational "
            "change must be reviewed and approved by authorized personnel."
        )

    return (
        f"For {airport}, the recommended first step is to verify the latest "
        "operational information relevant to the question. "
        + current_status
        + "Next, check the applicable rules, identify missing information, "
        "and assess whether escalation is needed. These indicators support "
        "scenario analysis only and cannot replace live operational data "
        "or human approval."
    )


def build_ai_context(data, airport, current_data):
    context_parts = []

    if current_data is not None and not current_data.empty:
        latest = current_data.sort_values("Date").iloc[-1]

        context_parts.append(
            "Selected airport: "
            + str(airport)
            + "\nLatest record date: "
            + str(latest["Date"].date())
            + "\nWOHI: "
            + str(round(float(latest["WOHI"]), 2))
            + "\nWeather risk index: "
            + str(round(float(latest["WeatherRisk"]), 2))
            + "\nAnomaly score: "
            + str(round(float(latest["AnomalyScore"]), 2))
            + "\nThese are derived from competition simulation data."
        )

    events = data.get("events", pd.DataFrame())

    if not events.empty:
        event_airport_col = find_column(
            events,
            ["机场名称", "机场", "Airport"]
        )

        if event_airport_col is not None:
            selected_events = events[
                events[event_airport_col].astype(str).str.strip()
                == str(airport)
            ]

            if not selected_events.empty:
                context_parts.append(
                    "Relevant simulated events:\n"
                    + selected_events.head(5).to_string(index=False)
                )

    rules = data.get("rules", pd.DataFrame())

    if not rules.empty:
        context_parts.append(
            "Competition operational rules:\n"
            + rules.head(12).to_string(index=False)
        )

    quality = data.get("quality", pd.DataFrame())

    if not quality.empty:
        context_parts.append(
            "Known data quality limitations:\n"
            + quality.head(6).to_string(index=False)
        )

    return "\n\n".join(context_parts)


def ask_qwen(
    question,
    airport,
    role,
    language,
    context,
    api_key,
    model_name
):
    if OpenAI is None:
        raise RuntimeError(
            "The openai package is not installed."
        )

    client = OpenAI(
        api_key=api_key.strip(),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=45,
        max_retries=1,
    )

    if language == "zh":
        language_instruction = "请使用简体中文回答。"
    else:
        language_instruction = "Respond in English."

    prompt = f"""
You are an airport operations decision support assistant for HackSHSID 2026 Track 3.

The supplied workbook contains simulated competition data, not verified live airport data.

Selected airport: {airport}
Target audience: {role}
User question: {question}

Data context:
{context}

Instructions:
{language_instruction}

Provide a practical recommendation with these parts:
1. The main takeaway.
2. The most important checks to perform.
3. Recommended next actions.
4. Missing information and escalation conditions.
5. What requires human approval.

Keep the answer readable and concise.

Use facts from the supplied context when available.
Clearly distinguish observed data from estimates and assumptions.
Do not invent real-time flight status, flight numbers, delays, cancellations,
weather forecasts, or operational restrictions.
Do not claim that any action has already been taken.
Do not claim a simulated score is a real delay probability.
Do not automatically authorize runway changes, cancellations, or other
safety-critical operational decisions.
Do not fabricate rule identifiers or citations.
If the data does not answer a question, state what information is missing.
"""

    response = client.chat.completions.create(
        model=model_name.strip() or "qwen-plus",
        messages=[
            {
                "role": "system",
                "content": (
                    "You provide cautious, evidence-based airport operations "
                    "decision support. You must distinguish simulated data "
                    "from verified real-world facts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
    )

    result = response.choices[0].message.content

    if not result or not result.strip():
        raise RuntimeError(
            "The AI service returned an empty response."
        )

    return result.strip()


def clean_text_for_speech(text):
    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```.*?```",
        " ",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"`([^`]*)`",
        r"\1",
        text
    )

    text = re.sub(
        r"!\[[^\]]*\]\([^)]*\)",
        " ",
        text
    )

    text = re.sub(
        r"\[([^\]]+)\]\([^)]*\)",
        r"\1",
        text
    )

    text = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s*",
        "",
        text
    )

    text = re.sub(
        r"(?m)^\s*[-*+]\s+",
        "",
        text
    )

    text = re.sub(
        r"(?m)^\s*\d+[.)]\s+",
        "",
        text
    )

    text = re.sub(
        r"(?m)^\s*>\s*",
        "",
        text
    )

    text = re.sub(
        r"(?<!\w)#[\w\-]+",
        " ",
        text
    )

    text = re.sub(
        r"\b(?:https?://|www\.)\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"[*_~|`]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Keep only letters, numbers, and whitespace for speech.
    # Preserve common Chinese and English characters.
    text = re.sub(
        r"[^\w\s\u3400-\u9fff]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"_+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def make_natural_speech(
    answer,
    airport,
    language,
    role,
    question,
    api_key="",
    model_name="qwen-plus",
):
    if not answer or not answer.strip():
        return ""

    if api_key.strip() and OpenAI is not None:
        try:
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=35,
                max_retries=1,
            )

            language_name = (
                "Simplified Chinese"
                if language == "zh"
                else "English"
            )

            prompt = f"""
Rewrite the airport recommendation below as a natural spoken briefing.

Language: {language_name}
Airport: {airport}
Audience: {role}
User question: {question}

Requirements:
Write one coherent paragraph that sounds like a knowledgeable human speaking
to a colleague.

Do not read the original recommendation word for word.
Do not copy the interface structure or reproduce a list of dashboard metrics.
Explain the main takeaway, why it matters, and what should happen next.
Use natural transitions.
Keep it concise and easy to listen to.
Preserve important facts, uncertainty, and safety restrictions.
Never invent live flight information, delays, cancellations, or confirmed impacts.
Do not claim any action has already been taken.
Do not include headings, numbered lists, hashtags, Markdown, URLs, or citations.
Output only the spoken paragraph.

Original recommendation:
{answer}
"""

            response = client.chat.completions.create(
                model=model_name.strip() or "qwen-plus",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write natural spoken briefings that preserve "
                            "the source facts and uncertainty. Never invent "
                            "operational facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.6,
            )

            result = response.choices[0].message.content

            if result and result.strip():
                return clean_text_for_speech(result)

        except Exception:
            pass

    if language == "zh":
        if role == "旅客":
            spoken = (
                f"关于{airport}目前的情况，最重要的是先确认航班的最新状态。"
                "现有信息还不能证明某个航班一定会延误或取消。"
                "建议大家留意航空公司的官方通知，如果行程有变化，"
                "可以联系航空公司或机场服务人员进一步确认。"
            )

        elif role == "机场协调人员":
            spoken = (
                f"关于{airport}，我们现在应该先确认最新天气预警和它的有效时段。"
                "接下来需要和航空公司核对航班计划、飞机后续执行安排、"
                "机组和地面保障状态，同时确认跑道和机位的实际情况。"
                "如果发现运行条件达到正式规则规定的升级标准，"
                "就应及时交由负责的运行人员评估。"
                "目前这些信息还不足以确定航班一定会受到影响，"
                "任何实际调整都需要由授权人员决定。"
            )

        else:
            spoken = (
                f"对于{airport}，我建议先把重点放在天气和现场运行条件的核实上。"
                "第一步是确认最新天气预警及其有效时间，"
                "然后检查航班计划、飞机后续执行安排、机组和地面保障情况。"
                "同时也要确认跑道、机位和机场容量是否受到限制。"
                "如果核查后发现情况符合正式运营规则的升级条件，"
                "再交由相关运行负责人处理。"
                "目前的数据只能帮助我们识别风险，"
                "还不能据此断定航班一定延误或取消，"
                "实际运行措施仍然需要授权人员审批。"
            )

    else:
        if role == "Passengers":
            spoken = (
                f"For {airport}, the most important thing is to check the latest "
                "official flight information before making travel decisions. "
                "The available information does not confirm that any flight "
                "will be delayed or cancelled. Please follow updates from your "
                "airline and contact the airline or airport service desk if "
                "your plans change."
            )

        elif role == "Airport Coordination":
            spoken = (
                f"For {airport}, the first priority is to verify the latest "
                "weather warning and how long it remains in effect. The team "
                "should then check flight schedules, aircraft rotations, crew "
                "availability, ground handling, and runway and stand capacity. "
                "If those checks reveal conditions that meet the formal escalation "
                "criteria, the responsible operations team should review the "
                "situation. The available information does not establish that "
                "flights will definitely be disrupted, and any operational changes "
                "must be approved by authorized personnel."
            )

        else:
            spoken = (
                f"For {airport}, I would start by checking the latest weather "
                "warning and confirming its effective period. Next, the team "
                "should verify flight schedules, aircraft rotations, crew "
                "availability, ground handling, and runway and stand capacity. "
                "If those checks identify a condition that meets the formal "
                "escalation criteria, the operations team should review it "
                "promptly. At this stage, the available data helps identify "
                "potential risks, but it does not prove that a flight will be "
                "delayed or cancelled. Any actual operational change still "
                "requires approval from authorized personnel."
            )

    return clean_text_for_speech(spoken)


async def generate_speech_bytes(text, voice):
    if edge_tts is None:
        raise RuntimeError(
            "edge-tts is not installed."
        )

    cleaned = clean_text_for_speech(text)

    if not cleaned:
        raise ValueError(
            "There is no readable speech content."
        )

    communicate = edge_tts.Communicate(
        text=cleaned,
        voice=voice
    )

    audio_buffer = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_buffer.seek(0)

    return audio_buffer.read()


def train_forecast_model(frame):
    work = frame.copy()

    work = work.sort_values(
        ["Airport", "Date"]
    ).reset_index(drop=True)

    work["MonthNumber"] = work["Date"].dt.month
    work["YearNumber"] = work["Date"].dt.year

    work["Lag1"] = work.groupby("Airport")["WOHI"].shift(1)
    work["Lag2"] = work.groupby("Airport")["WOHI"].shift(2)

    work["RollingMean3"] = work.groupby("Airport")["WOHI"].transform(
        lambda series: series.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )

    work["RollingStd3"] = work.groupby("Airport")["WOHI"].transform(
        lambda series: series.shift(1).rolling(
            3,
            min_periods=2
        ).std()
    )

    work["RollingStd3"] = work["RollingStd3"].fillna(0)

    feature_columns = [
        "YearNumber",
        "MonthNumber",
        "WeatherRisk",
        "Lag1",
        "Lag2",
        "RollingMean3",
        "RollingStd3",
    ]

    work = work.dropna(
        subset=["WOHI", "Lag1"]
    ).copy()

    if len(work) < 12:
        return {
            "trained": False,
            "message": (
                "There are not enough historical records to train "
                "a useful demonstration model."
            ),
        }

    work[feature_columns] = work[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    work[feature_columns] = work[feature_columns].fillna(
        work[feature_columns].median()
    )

    X = work[feature_columns]
    y = work["WOHI"]

    split = max(
        int(len(work) * 0.8),
        1
    )

    if len(work) - split < 1:
        split = len(work) - 1

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    if len(X_train) < 5 or len(X_test) < 1:
        return {
            "trained": False,
            "message": "The training split is too small.",
        }

    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # Fit a final model using all available historical records.
    final_model = RandomForestRegressor(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(X, y)

    return {
        "trained": True,
        "model": final_model,
        "features": feature_columns,
        "mae": float(mae),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "data": work,
    }


def forecast_next_month(frame, airport, model_result):
    if not model_result or not model_result.get("trained"):
        return None

    airport_data = frame[
        frame["Airport"].astype(str) == str(airport)
    ].sort_values("Date").copy()

    if airport_data.empty:
        return None

    latest = airport_data.iloc[-1]

    history = airport_data["WOHI"].dropna().tolist()

    if not history:
        return None

    lag1 = history[-1]
    lag2 = history[-2] if len(history) >= 2 else lag1

    rolling_mean = float(
        np.mean(history[-3:])
    )

    rolling_std = (
        float(np.std(history[-3:], ddof=1))
        if len(history[-3:]) > 1
        else 0.0
    )

    next_date = latest["Date"] + pd.DateOffset(months=1)

    row = {
        "YearNumber": next_date.year,
        "MonthNumber": next_date.month,
        "WeatherRisk": float(latest["WeatherRisk"]),
        "Lag1": float(lag1),
        "Lag2": float(lag2),
        "RollingMean3": rolling_mean,
        "RollingStd3": rolling_std,
    }

    features = model_result["features"]

    X_next = pd.DataFrame(
        [[row[column] for column in features]],
        columns=features
    )

    prediction = model_result["model"].predict(X_next)[0]

    return {
        "date": next_date,
        "prediction": float(np.clip(prediction, 0, 100)),
        "last_observed": float(latest["WOHI"]),
    }


def run_monte_carlo(frame, airport, simulations=1500, seed=42):
    airport_data = frame[
        frame["Airport"].astype(str) == str(airport)
    ].sort_values("Date")

    if airport_data.empty:
        return None

    rng = np.random.default_rng(seed)

    historical_weather = airport_data["WeatherRisk"].dropna().to_numpy()

    if len(historical_weather) == 0:
        historical_weather = np.array([0.0])

    latest = airport_data.iloc[-1]

    baseline_wohi = float(latest["WOHI"])

    simulated_weather = rng.choice(
        historical_weather,
        size=simulations,
        replace=True
    )

    pressure = rng.beta(
        2.0,
        5.0,
        size=simulations
    )

    uncertainty = rng.normal(
        0,
        3.0,
        size=simulations
    )

    scenario_wohi = (
        baseline_wohi
        - simulated_weather * 0.08
        - pressure * 12.0
        + uncertainty
    )

    scenario_wohi = np.clip(
        scenario_wohi,
        0,
        100
    )

    return {
        "values": scenario_wohi,
        "median": float(np.median(scenario_wohi)),
        "p10": float(np.percentile(scenario_wohi, 10)),
        "p90": float(np.percentile(scenario_wohi, 90)),
        "below_70": float(np.mean(scenario_wohi < 70) * 100),
    }


def make_trend_chart(frame, airport, language):
    airport_data = frame[
        frame["Airport"].astype(str) == str(airport)
    ].sort_values("Date").tail(24).copy()

    if airport_data.empty:
        return None

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=airport_data["Date"],
            y=airport_data["WOHI"],
            mode="lines+markers",
            name="WOHI",
            line=dict(
                color="#39c5f6",
                width=3
            ),
            marker=dict(
                size=7,
                color="#39c5f6"
            ),
            hovertemplate=(
                "%{x|%b %Y}<br>"
                "WOHI: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )

    anomalies = airport_data[
        airport_data["AnomalyScore"].abs() >= 1.5
    ]

    if not anomalies.empty:
        figure.add_trace(
            go.Scatter(
                x=anomalies["Date"],
                y=anomalies["WOHI"],
                mode="markers",
                name=(
                    "Critical Anomaly"
                    if language == "en"
                    else "异常提示"
                ),
                marker=dict(
                    color="#ffbd45",
                    size=13,
                    symbol="x",
                    line=dict(width=2)
                ),
                hovertemplate=(
                    "%{x|%b %Y}<br>"
                    "WOHI: %{y:.1f}"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=80,
        line_dash="dot",
        line_color="rgba(255,255,255,0.42)",
        annotation_text=(
            "Reference level"
            if language == "en"
            else "参考水平"
        ),
        annotation_position="top left",
        annotation_font_color="#dce9f7",
    )

    figure.update_layout(
        height=390,
        autosize=True,
        margin=dict(
            l=45,
            r=22,
            t=20,
            b=55
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,20,36,0.65)",
        font=dict(
            color="#e6eef9",
            size=12
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#e6eef9"),
        ),
        xaxis=dict(
            title="Month" if language == "en" else "月份",
            showgrid=False,
            zeroline=False,
            automargin=True,
            tickformat="%b %Y",
            tickangle=0,
            fixedrange=False,
        ),
        yaxis=dict(
            title="WOHI",
            range=[0, 105],
            dtick=20,
            gridcolor="rgba(220,235,250,0.16)",
            zeroline=False,
            automargin=True,
            fixedrange=False,
        ),
        hovermode="x unified",
    )

    return figure


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def translate_role(role, language):
    if language == "zh":
        mapping = {
            "Operations Staff": "运营人员",
            "Airport Coordination": "机场协调人员",
            "Passengers": "旅客",
        }
    else:
        mapping = {
            "Operations Staff": "Operations Staff",
            "Airport Coordination": "Airport Coordination",
            "Passengers": "Passengers",
        }

    return mapping.get(role, role)


def main():
    with st.sidebar:
        st.markdown("## ✈️ Control Center")

        language_choice = st.radio(
            "Language / 语言",
            ["English", "中文"],
            horizontal=True,
            key="language_choice",
        )

        language = (
            "zh"
            if language_choice == "中文"
            else "en"
        )

        st.divider()

        st.markdown("### Data Source")

        uploaded_file = st.file_uploader(
            "Upload Track 3 Excel workbook",
            type=["xlsx", "xls"],
            help=(
                "Upload the competition workbook. "
                "If it is in the same folder as app.py, "
                "the app can also try to load it automatically."
            ),
        )

        local_path = st.text_input(
            "Local workbook path",
            value=DEFAULT_EXCEL,
        )

        st.divider()

        st.markdown("### AI Configuration")

        api_key = st.text_input(
            "DashScope API Key",
            value=os.getenv("DASHSCOPE_API_KEY", ""),
            type="password",
            help="Your key is used for the current session.",
        )

        model_name = st.text_input(
            "Model",
            value="qwen-plus",
        )

        online_ai = bool(
            api_key.strip()
            and OpenAI is not None
        )

        if online_ai:
            st.success("Online AI enabled")
        else:
            st.info(
                "Local fallback is available without an API key."
            )

        st.divider()

        st.markdown("### About the model")

        st.caption(
            "The workbook contains simulated competition data. "
            "Predictions and scenario outputs are decision support "
            "demonstrations, not live operational forecasts."
        )

    try:
        data = load_uploaded_or_local(
            uploaded_file,
            local_path
        )
    except Exception as exc:
        data = None
        load_error = str(exc)
    else:
        load_error = ""

    st.markdown(
        """
        <div class="hero">
            <div class="section-label">HackSHSID 2026 · Track 3</div>
            <div class="hero-title">
                Airport Operations Intelligence
            </div>
            <div class="hero-subtitle">
                Operational health monitoring, risk analysis,
                and human-supervised decision support
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data is None:
        st.markdown(
            """
            <div class="notice">
                The Excel workbook has not been loaded yet.
                Upload the Track 3 workbook in the sidebar, or put it
                in the same folder as app.py and check the local path.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if load_error:
            st.error(
                "Workbook loading failed. "
                "Check the sheet names and column headers. "
                f"Details: {load_error}"
            )

        st.stop()

    baseline = data["baseline"]

    airports = sorted(
        baseline["Airport"].dropna().astype(str).unique().tolist()
    )

    if not airports:
        st.error("No airport names were found in the workbook.")
        st.stop()

    with st.sidebar:
        selected_airport = st.selectbox(
            "Select airport",
            airports,
            key="selected_airport",
        )

        if language == "zh":
            role_options = [
                "运营人员",
                "机场协调人员",
                "旅客",
            ]
            role_display = st.radio(
                "目标受众",
                role_options,
                index=0,
            )
            role_map = {
                "运营人员": "Operations Staff",
                "机场协调人员": "Airport Coordination",
                "旅客": "Passengers",
            }
            role = role_map[role_display]
        else:
            role = st.radio(
                "Target audience",
                [
                    "Operations Staff",
                    "Airport Coordination",
                    "Passengers",
                ],
                index=0,
            )

        if st.button(
            "Reset Assistant" if language == "en" else "重置助手",
            key="reset_assistant",
        ):
            st.session_state.assistant_answer = ""
            st.session_state.last_question = ""
            st.session_state.natural_speech = ""
            st.session_state.conversation = []
            st.rerun()

    airport_data = get_latest_airport_data(
        data,
        selected_airport
    )

    latest = (
        airport_data.iloc[-1]
        if not airport_data.empty
        else None
    )

    st.markdown(
        '<div class="section-label">AI COMMAND CENTER</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Ask the Operations Assistant"
            if language == "en"
            else "### 询问机场运营智能助手"
        )

        st.caption(
            "Get a contextual recommendation based on the selected airport, "
            "competition workbook, and operational rules."
            if language == "en"
            else "根据所选机场、赛事数据和运营规则生成建议。"
        )

        with st.form("question_form", clear_on_submit=False):
            question = st.text_area(
                "Enter your question"
                if language == "en"
                else "请输入你的问题",
                value=(
                    "What are our top priorities?"
                    if language == "en"
                    else "我们当前最重要的优先事项是什么？"
                ),
                height=105,
                key="question_input",
            )

            submitted = st.form_submit_button(
                "Generate Recommendation"
                if language == "en"
                else "生成运营建议",
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    if submitted:
        cleaned_question = question.strip()

        if not cleaned_question:
            st.warning(
                "Please enter a question."
                if language == "en"
                else "请先输入问题。"
            )
        else:
            st.session_state.last_question = cleaned_question
            st.session_state.natural_speech = ""

            context = build_ai_context(
                data,
                selected_airport,
                airport_data,
            )

            fallback = make_fallback_answer(
                question=cleaned_question,
                airport=selected_airport,
                role=role,
                language=language,
                current_data=airport_data,
                rules=data["rules"],
            )

            if online_ai:
                with st.spinner(
                    "Generating recommendation..."
                    if language == "en"
                    else "正在生成建议..."
                ):
                    try:
                        answer = ask_qwen(
                            question=cleaned_question,
                            airport=selected_airport,
                            role=role,
                            language=language,
                            context=context,
                            api_key=api_key,
                            model_name=model_name,
                        )

                        st.session_state.assistant_answer = answer

                    except Exception as exc:
                        st.session_state.assistant_answer = fallback

                        st.warning(
                            (
                                "The online AI request failed. "
                                "A local recommendation is shown instead. "
                                f"Details: {exc}"
                            )
                            if language == "en"
                            else (
                                "在线 AI 请求失败，已切换到本地备用建议。"
                                f"错误信息：{exc}"
                            )
                        )
            else:
                st.session_state.assistant_answer = fallback

            st.session_state.conversation.append(
                {
                    "question": cleaned_question,
                    "answer": st.session_state.assistant_answer,
                }
            )

    if st.session_state.assistant_answer:
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### AI Recommendation"
            if language == "en"
            else "### AI 运营建议"
        )

        st.markdown(
            '<div class="answer-box">'
            + st.session_state.assistant_answer.replace(
                "&", "&amp;"
            ).replace(
                "<", "&lt;"
            ).replace(
                ">", "&gt;"
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-label">OPERATIONAL OVERVIEW</div>',
        unsafe_allow_html=True,
    )

    if latest is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card(
                "WOHI",
                f"{latest['WOHI']:.1f}",
                (
                    "Operational health index"
                    if language == "en"
                    else "运行健康指数"
                ),
            )

        with col2:
            metric_card(
                "Weather Risk",
                f"{latest['WeatherRisk']:.1f}",
                (
                    "Historical risk indicator"
                    if language == "en"
                    else "历史天气风险指标"
                ),
            )

        with col3:
            anomaly = abs(
                float(latest["AnomalyScore"])
            )

            metric_card(
                "Anomaly",
                (
                    "Review"
                    if anomaly >= 1.5
                    else "Normal"
                ),
                (
                    f"Absolute score {anomaly:.2f}"
                    if language == "en"
                    else f"异常分数绝对值 {anomaly:.2f}"
                ),
            )

        with col4:
            metric_card(
                "Records",
                str(len(airport_data)),
                (
                    "Monthly observations"
                    if language == "en"
                    else "月度记录数"
                ),
            )

    chart_col, info_col = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with chart_col:
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Operational Health Trend"
            if language == "en"
            else "### 运行健康趋势"
        )

        chart = make_trend_chart(
            baseline,
            selected_airport,
            language,
        )

        if chart is not None:
            st.plotly_chart(
                chart,
                use_container_width=True,
                config={
                    "responsive": True,
                    "displayModeBar": False,
                    "scrollZoom": False,
                },
            )
        else:
            st.info(
                "No trend data is available."
                if language == "en"
                else "暂无趋势数据。"
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with info_col:
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Airport Snapshot"
            if language == "en"
            else "### 机场概况"
        )

        if latest is not None:
            st.write(
                (
                    f"**Airport:** {selected_airport}"
                    if language == "en"
                    else f"**机场：** {selected_airport}"
                )
            )

            st.write(
                (
                    f"**Latest record:** {latest['Date'].strftime('%B %Y')}"
                    if language == "en"
                    else f"**最新记录：** {latest['Date'].strftime('%Y年%m月')}"
                )
            )

            st.write(
                (
                    f"**Airport category:** {latest['AirportType']}"
                    if language == "en"
                    else f"**机场类别：** {latest['AirportType']}"
                )
            )

            st.write(
                (
                    f"**Schedule completion:** "
                    f"{latest['ScheduleRate'] * 100:.1f}%"
                    if language == "en"
                    else f"**计划完成比例：** "
                    f"{latest['ScheduleRate'] * 100:.1f}%"
                )
            )

            st.write(
                (
                    f"**Normal release rate:** "
                    f"{latest['ReleaseRate'] * 100:.1f}%"
                    if language == "en"
                    else f"**正常放行比例：** "
                    f"{latest['ReleaseRate'] * 100:.1f}%"
                )
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-label">PREDICTIVE ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    forecast_col, simulation_col = st.columns(
        2,
        gap="large",
    )

    with forecast_col:
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Machine Learning Forecast"
            if language == "en"
            else "### 机器学习预测"
        )

        if st.button(
            "Train and Forecast"
            if language == "en"
            else "训练模型并预测",
            key="train_forecast",
        ):
            with st.spinner(
                "Training the Random Forest model..."
                if language == "en"
                else "正在训练随机森林模型..."
            ):
                result = train_forecast_model(baseline)
                st.session_state.model_result = result

        model_result = st.session_state.model_result

        if model_result is None:
            st.info(
                "Train a model to see a next-month demonstration forecast."
                if language == "en"
                else "点击按钮训练模型并查看下个月的示例预测。"
            )

        elif not model_result.get("trained"):
            st.warning(
                model_result.get(
                    "message",
                    "The model could not be trained."
                )
            )

        else:
            forecast = forecast_next_month(
                baseline,
                selected_airport,
                model_result,
            )

            if forecast is None:
                st.warning(
                    "Not enough airport-specific data."
                    if language == "en"
                    else "该机场的数据不足，暂时无法预测。"
                )
            else:
                st.metric(
                    (
                        f"Forecast WOHI for {forecast['date'].strftime('%b %Y')}"
                        if language == "en"
                        else f"{forecast['date'].strftime('%Y年%m月')}预测 WOHI"
                    ),
                    f"{forecast['prediction']:.1f}",
                    delta=(
                        f"{forecast['prediction'] - forecast['last_observed']:+.1f}"
                    ),
                )

                st.write(
                    (
                        f"Backtest mean absolute error: "
                        f"{model_result['mae']:.2f} points"
                        if language == "en"
                        else f"历史测试集平均绝对误差："
                        f"{model_result['mae']:.2f} 分"
                    )
                )

                st.caption(
                    (
                        "The model uses time-ordered training and testing. "
                        "A forecast is a demonstration based on historical "
                        "simulated data, not a live airport prediction."
                        if language == "en"
                        else
                        "模型按时间顺序划分训练集和测试集。"
                        "预测仅基于历史模拟数据，不代表真实机场预测。"
                    )
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with simulation_col:
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Monte Carlo Scenario Analysis"
            if language == "en"
            else "### 蒙特卡洛情景分析"
        )

        simulations = st.slider(
            "Number of simulations"
            if language == "en"
            else "模拟次数",
            min_value=300,
            max_value=3000,
            value=1500,
            step=300,
        )

        if st.button(
            "Run Scenario Simulation"
            if language == "en"
            else "运行情景模拟",
            key="run_monte_carlo",
        ):
            with st.spinner(
                "Simulating possible scenarios..."
                if language == "en"
                else "正在模拟可能的情景..."
            ):
                st.session_state["monte_carlo_result"] = run_monte_carlo(
                    baseline,
                    selected_airport,
                    simulations=simulations,
                )

        mc = st.session_state.get(
            "monte_carlo_result"
        )

        if mc is None:
            st.info(
                "Run the simulation to examine hypothetical health index scenarios."
                if language == "en"
                else "运行模拟以查看假设的运行健康指数情景。"
            )

        else:
            st.metric(
                (
                    "Median scenario WOHI"
                    if language == "en"
                    else "情景 WOHI 中位数"
                ),
                f"{mc['median']:.1f}",
            )

            st.write(
                (
                    f"Central scenario range: "
                    f"{mc['p10']:.1f} to {mc['p90']:.1f}"
                    if language == "en"
                    else f"情景第十至第九十百分位范围："
                    f"{mc['p10']:.1f} 至 {mc['p90']:.1f}"
                )
            )

            st.write(
                (
                    f"Share of simulated scenarios below 70: "
                    f"{mc['below_70']:.1f}%"
                    if language == "en"
                    else f"模拟情景中 WOHI 低于 70 的比例："
                    f"{mc['below_70']:.1f}%"
                )
            )

            simulation_figure = go.Figure(
                data=[
                    go.Histogram(
                        x=mc["values"],
                        nbinsx=28,
                        marker_color="#39c5f6",
                        opacity=0.85,
                    )
                ]
            )

            simulation_figure.update_layout(
                height=280,
                autosize=True,
                margin=dict(
                    l=38,
                    r=15,
                    t=10,
                    b=45,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(5,20,36,0.65)",
                font=dict(color="#e6eef9"),
                xaxis=dict(
                    title="Scenario WOHI",
                    gridcolor="rgba(220,235,250,0.12)",
                    automargin=True,
                ),
                yaxis=dict(
                    title="Count",
                    gridcolor="rgba(220,235,250,0.12)",
                    automargin=True,
                ),
                showlegend=False,
            )

            st.plotly_chart(
                simulation_figure,
                use_container_width=True,
                config={
                    "responsive": True,
                    "displayModeBar": False,
                },
            )

        st.caption(
            (
                "This is a hypothetical sensitivity simulation using "
                "historical weather-risk values and assumed operating pressure. "
                "It is not a calibrated probability of flight delay."
                if language == "en"
                else
                "这是基于历史天气风险和假设运行压力的敏感性模拟，"
                "不是经过校准的航班延误概率。"
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-label">OPERATIONAL RULES AND EVENTS</div>',
        unsafe_allow_html=True,
    )

    rules_col, events_col = st.columns(
        2,
        gap="large",
    )

    with rules_col:
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Operational Rules"
            if language == "en"
            else "### 运营规则"
        )

        if data["rules"].empty:
            st.info(
                "No operational rules were found."
                if language == "en"
                else "没有找到运营规则。"
            )
        else:
            st.dataframe(
                data["rules"],
                use_container_width=True,
                hide_index=True,
                height=300,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with events_col:
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Simulated Events"
            if language == "en"
            else "### 模拟事件"
        )

        if data["events"].empty:
            st.info(
                "No simulated events were found."
                if language == "en"
                else "没有找到模拟事件。"
            )
        else:
            event_airport_col = find_column(
                data["events"],
                ["机场名称", "机场", "Airport"]
            )

            if event_airport_col is not None:
                selected_events = data["events"][
                    data["events"][event_airport_col].astype(str).str.strip()
                    == selected_airport
                ]

                if selected_events.empty:
                    selected_events = data["events"]

            else:
                selected_events = data["events"]

            st.dataframe(
                selected_events,
                use_container_width=True,
                hide_index=True,
                height=300,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-label">NATURAL SPEECH GENERATION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="glass">',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Spoken Briefing"
        if language == "en"
        else "### 自然口语简报"
    )

    st.caption(
        "Generate a separate conversational paragraph for audio. "
        "It will summarize the recommendation naturally instead of "
        "reading the interface word for word."
        if language == "en"
        else
        "生成单独的自然口语稿，用于语音播放。"
        "它会重新组织重点，而不是逐字朗读界面内容。"
    )

    if not st.session_state.assistant_answer:
        st.info(
            "Generate an AI recommendation first."
            if language == "en"
            else "请先生成一份运营建议。"
        )

    else:
        if st.button(
            "Prepare Natural Speech"
            if language == "en"
            else "生成自然口语稿",
            key="prepare_natural_speech",
        ):
            with st.spinner(
                "Preparing a conversational briefing..."
                if language == "en"
                else "正在生成自然口语简报..."
            ):
                st.session_state.natural_speech = make_natural_speech(
                    answer=st.session_state.assistant_answer,
                    airport=selected_airport,
                    language=language,
                    role=role,
                    question=st.session_state.last_question,
                    api_key=api_key if online_ai else "",
                    model_name=model_name,
                )

        if st.session_state.natural_speech:
            st.text_area(
                "Speech script"
                if language == "en"
                else "口语稿",
                value=st.session_state.natural_speech,
                height=145,
                key="speech_preview",
            )

            if edge_tts is None:
                st.warning(
                    "Install edge-tts to enable audio generation."
                    if language == "en"
                    else "请安装 edge-tts 以启用语音生成。"
                )

            else:
                if st.button(
                    "Generate Audio"
                    if language == "en"
                    else "生成语音",
                    key="generate_natural_audio",
                ):
                    try:
                        cleaned = clean_text_for_speech(
                            st.session_state.natural_speech
                        )

                        if not cleaned:
                            st.warning(
                                "There is no readable speech content."
                                if language == "en"
                                else "没有可朗读的内容。"
                            )

                        else:
                            voice = (
                                "zh-CN-XiaoxiaoNeural"
                                if language == "zh"
                                else "en-US-JennyNeural"
                            )

                            with st.spinner(
                                "Generating audio..."
                                if language == "en"
                                else "正在生成语音..."
                            ):
                                audio_bytes = asyncio.run(
                                    generate_speech_bytes(
                                        cleaned,
                                        voice,
                                    )
                                )

                            if audio_bytes:
                                st.audio(
                                    audio_bytes,
                                    format="audio/mp3",
                                )
                            else:
                                st.error(
                                    "No audio was returned."
                                    if language == "en"
                                    else "语音服务没有返回音频。"
                                )

                    except Exception as exc:
                        st.error(
                            f"Speech generation failed: {exc}"
                        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">DATA QUALITY AND TRANSPARENCY</div>',
        unsafe_allow_html=True,
    )

    quality_col, feedback_col = st.columns(
        2,
        gap="large",
    )

    with quality_col:
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Data Quality"
            if language == "en"
            else "### 数据质量"
        )

        if data["quality"].empty:
            st.info(
                "No separate data quality issue table was found."
                if language == "en"
                else "没有找到单独的数据质量问题表。"
            )
        else:
            st.dataframe(
                data["quality"],
                use_container_width=True,
                hide_index=True,
                height=230,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with feedback_col:
        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "### Human Feedback"
            if language == "en"
            else "### 人工反馈"
        )

        if data["feedback"].empty:
            st.info(
                "No human feedback records were found."
                if language == "en"
                else "没有找到人工反馈记录。"
            )
        else:
            st.dataframe(
                data["feedback"],
                use_container_width=True,
                hide_index=True,
                height=230,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="notice">
            Decision support only. All workbook values are competition
            simulation data. The model does not access live flight
            operations, and no safety-critical operational action is
            automatically executed. Authorized personnel must verify
            information and approve actual operational decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(
        "View raw monthly records"
        if language == "en"
        else "查看月度原始记录"
    ):
        st.dataframe(
            airport_data,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "HackSHSID 2026 Track 3 · Airport Operations Decision Support"
    )


if __name__ == "__main__":
    main()
