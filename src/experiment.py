import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# НАСТРОЙКИ ПРОЕКТА
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"

KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
RESULTS_PATH = RESULTS_DIR / "raw_results.json"


# =========================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# =========================================================

MODEL_NAME = "deepseek-chat"

# Небольшая задержка между запросами.
# Нужна, чтобы не отправлять запросы подряд без паузы.
REQUEST_DELAY = 1.0


# =========================================================
# DEEPSEEK API
# =========================================================

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "Не найден DEEPSEEK_API_KEY. "
        "Проверь файл .env."
    )


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# =========================================================
# ЗАГРУЗКА БАЗЫ ЗНАНИЙ
# =========================================================

def load_knowledge_base() -> dict:
    """Загружает тестовую базу знаний."""

    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def knowledge_base_to_text(data: dict) -> str:
    """Преобразует JSON-базу знаний в текст для LLM."""

    organization = data["organization"]
    departments = data["departments"]
    technology = data["technology"]
    infrastructure = data["infrastructure"]
    helpdesk = data["helpdesk"]

    lines = [
        "ОРГАНИЗАЦИЯ",
        f"Название: {organization['name']}",
        f"Год основания: {organization['founded']}",
        f"Город: {organization['city']}",
        f"Количество сотрудников: {organization['employees']}",
        "",
        "ПОДРАЗДЕЛЕНИЯ"
    ]

    for department in departments:
        lines.append(
            f"Название: {department['name']}"
        )

        if "employees" in department:
            lines.append(
                f"Количество сотрудников: "
                f"{department['employees']}"
            )

        if "head" in department:
            lines.append(
                f"Руководитель: "
                f"{department['head']}"
            )

        lines.append("")

    lines.extend([
        "ТЕХНОЛОГИИ",
        f"Основной язык программирования: "
        f"{technology['main_programming_language']}",

        f"Система управления базами данных: "
        f"{technology['database']}",

        f"WebSocket: "
        f"{'используется' if technology['websocket'] else 'не используется'}",

        "",
        "ИНФРАСТРУКТУРА",

        f"Резервное копирование: "
        f"{infrastructure['backup']['frequency']}",

        f"Время резервного копирования: "
        f"{infrastructure['backup']['time']}",

        f"Срок хранения резервных копий: "
        f"{infrastructure['backup']['retention_days']} дней",

        f"Аутентификация: "
        f"{infrastructure['authentication']}",

        f"MFA: "
        f"{'включена' if infrastructure['multi_factor_authentication']['enabled'] else 'отключена'}",

        f"MFA применяется к: "
        f"{infrastructure['multi_factor_authentication']['applies_to']}",

        "",
        "HELPDESK",

        f"Используется: "
        f"{'да' if helpdesk['used'] else 'нет'}",

        f"Количество уровней приоритета: "
        f"{helpdesk['priority_levels']}",

        f"Наивысший приоритет: "
        f"{helpdesk['highest_priority']}",

        f"Уведомления отправляются: "
        f"{helpdesk['notifications']}"
    ])

    return "\n".join(lines)


# =========================================================
# ЗАГРУЗКА ВОПРОСОВ
# =========================================================

def load_questions() -> list[dict]:
    """Загружает набор экспериментальных вопросов."""

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# ЗАПРОС К DEEPSEEK
# =========================================================

def ask_deepseek(
    question: str,
    system_prompt: str
) -> str:
    """Отправляет один запрос в DeepSeek."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    if answer is None:
        return ""

    return answer.strip()


# =========================================================
# ПРОМПТЫ ЭКСПЕРИМЕНТА
# =========================================================

def build_baseline_prompt(
    knowledge_text: str
) -> str:
    """
    Базовый режим.

    Модель получает базу знаний, но не получает
    специальных инструкций по предотвращению галлюцинаций.
    """

    return f"""
Ты являешься интеллектуальным помощником.

Ответь на вопрос пользователя.

База знаний:

{knowledge_text}
"""


def build_prompt_controlled(
    knowledge_text: str
) -> str:
    """
    Контролируемый режим.

    Добавляются специальные инструкции,
    направленные на снижение вероятности галлюцинаций.
    """

    return f"""
Ты являешься интеллектуальным помощником
корпоративной информационной системы.

Используй ТОЛЬКО предоставленную базу знаний.

Правила:

1. Не используй информацию, которой нет в базе знаний.
2. Не придумывай отсутствующие сведения.
3. Если информация отсутствует, прямо сообщи об этом.
4. Если вопрос содержит ложную предпосылку,
   укажи на это.
5. Не используй свои общие знания для дополнения базы.
6. Если в вопросе предлагается несколько вариантов,
   выбирай только тот вариант, который подтверждается
   базой знаний.
7. Если ни один вариант не подтверждается базой знаний,
   сообщи, что определить ответ невозможно.

База знаний:

{knowledge_text}
"""


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def current_timestamp() -> str:
    """Возвращает время выполнения запроса в UTC."""

    return datetime.now(timezone.utc).isoformat()


def save_results(results: list[dict]) -> None:
    """Сохраняет результаты эксперимента в JSON."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )


def print_separator() -> None:
    print("-" * 70)


# =========================================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# =========================================================

def main():

    print("=" * 70)
    print("ЭКСПЕРИМЕНТ ПО ИССЛЕДОВАНИЮ ГАЛЛЮЦИНАЦИЙ LLM")
    print("=" * 70)

    # -----------------------------------------------------
    # Загрузка данных
    # -----------------------------------------------------

    print("\nЗагрузка базы знаний...")

    knowledge_base = load_knowledge_base()

    knowledge_text = knowledge_base_to_text(
        knowledge_base
    )

    print("База знаний загружена.")

    print("\nЗагрузка вопросов...")

    questions = load_questions()

    print(
        f"Загружено вопросов: {len(questions)}"
    )

    # -----------------------------------------------------
    # Формирование промптов
    # -----------------------------------------------------

    baseline_prompt = build_baseline_prompt(
        knowledge_text
    )

    controlled_prompt = build_prompt_controlled(
        knowledge_text
    )

    # -----------------------------------------------------
    # Результаты
    # -----------------------------------------------------

    results = []

    total_requests = len(questions) * 2

    current_request = 0

    print("\n")
    print("=" * 70)
    print(
        f"НАЧАЛО ЭКСПЕРИМЕНТА: "
        f"{len(questions)} вопросов × 2 режима"
    )
    print(
        f"Всего запросов к модели: {total_requests}"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Перебор вопросов
    # -----------------------------------------------------

    for index, question_data in enumerate(
        questions,
        start=1
    ):

        question_id = question_data["id"]
        category = question_data["category"]
        question = question_data["question"]
        expected = question_data.get("expected")

        print("\n")
        print("=" * 70)

        print(
            f"ВОПРОС {index}/{len(questions)}"
        )

        print(
            f"ID: {question_id}"
        )

        print(
            f"Категория: {category}"
        )

        print(
            f"Вопрос: {question}"
        )

        print("=" * 70)

        # =================================================
        # BASELINE
        # =================================================

        current_request += 1

        print(
            f"\n[{current_request}/{total_requests}] "
            f"BASELINE"
        )

        try:

            start_time = time.perf_counter()

            baseline_answer = ask_deepseek(
                question,
                baseline_prompt
            )

            elapsed = time.perf_counter() - start_time

            print("\nОтвет:")

            print(baseline_answer)

            results.append({
                "question_id": question_id,
                "category": category,
                "question": question,
                "expected": expected,
                "mode": "baseline",
                "answer": baseline_answer,
                "timestamp": current_timestamp(),
                "response_time_seconds": round(
                    elapsed,
                    3
                ),
                "error": None
            })

        except Exception as error:

            error_text = str(error)

            print(
                f"\nОШИБКА BASELINE: "
                f"{error_text}"
            )

            results.append({
                "question_id": question_id,
                "category": category,
                "question": question,
                "expected": expected,
                "mode": "baseline",
                "answer": "",
                "timestamp": current_timestamp(),
                "response_time_seconds": None,
                "error": error_text
            })

        save_results(results)

        time.sleep(REQUEST_DELAY)

        # =================================================
        # PROMPT
        # =================================================

        current_request += 1

        print(
            f"\n[{current_request}/{total_requests}] "
            f"PROMPT"
        )

        try:

            start_time = time.perf_counter()

            controlled_answer = ask_deepseek(
                question,
                controlled_prompt
            )

            elapsed = time.perf_counter() - start_time

            print("\nОтвет:")

            print(controlled_answer)

            results.append({
                "question_id": question_id,
                "category": category,
                "question": question,
                "expected": expected,
                "mode": "prompt",
                "answer": controlled_answer,
                "timestamp": current_timestamp(),
                "response_time_seconds": round(
                    elapsed,
                    3
                ),
                "error": None
            })

        except Exception as error:

            error_text = str(error)

            print(
                f"\nОШИБКА PROMPT: "
                f"{error_text}"
            )

            results.append({
                "question_id": question_id,
                "category": category,
                "question": question,
                "expected": expected,
                "mode": "prompt",
                "answer": "",
                "timestamp": current_timestamp(),
                "response_time_seconds": None,
                "error": error_text
            })

        save_results(results)

        # -------------------------------------------------
        # Пауза перед следующим вопросом
        # -------------------------------------------------

        if index < len(questions):

            time.sleep(REQUEST_DELAY)

    # =====================================================
    # СОХРАНЕНИЕ
    # =====================================================

    save_results(results)

    # =====================================================
    # ИТОГ
    # =====================================================

    print("\n")
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 70)

    print(
        f"\nПолучено результатов: "
        f"{len(results)}"
    )

    print(
        f"Ожидалось результатов: "
        f"{total_requests}"
    )

    print(
        f"\nФайл с исходными результатами:"
    )

    print(
        RESULTS_PATH
    )

    print("\nРезультаты сохранены.")


# =========================================================
# ЗАПУСК
# =========================================================

if __name__ == "__main__":
    main()