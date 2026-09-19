# src/evaluator.py

import json
import re
from pathlib import Path
from collections import Counter

from ground_truth import (
    GROUND_TRUTH,
    get_ground_truth,
    validate_ground_truth,
)


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_RESULTS_FILE = (
    BASE_DIR
    / "data"
    / "results"
    / "raw_results.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "results"
    / "evaluated_results.json"
)


# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def load_json(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def normalize(text: str) -> str:

    text = str(text).lower()

    text = text.replace("ё", "е")
    text = text.replace("«", "")
    text = text.replace("»", "")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def contains(text: str, values) -> bool:

    text = normalize(text)

    for value in values:

        if normalize(value) in text:
            return True

    return False


# =========================================================
# ОПРЕДЕЛЕНИЕ ОТКАЗА
# =========================================================

REFUSAL_PATTERNS = [

    "нет информации",
    "нет данных",
    "данные отсутствуют",
    "сведения отсутствуют",
    "информация отсутствует",

    "не указано",
    "не указан",
    "не указана",
    "не указаны",

    "не сообщается",

    "невозможно определить",
    "невозможно установить",

    "нельзя определить",
    "нельзя установить",

    "нельзя утверждать",
    "невозможно утверждать",

    "нельзя сказать",
    "невозможно сказать",

    "не позволяет утверждать",
    "не позволяет сделать вывод",

    "не следует из базы знаний",
    "не следует из предоставленной информации",

    "не содержится в базе знаний",
    "отсутствует в базе знаний",

    "в базе знаний нет",
    "в базе знаний отсутствует",

    "база знаний не содержит",

    "недостаточно данных",
    "недостаточно информации",

    "невозможно сделать однозначный вывод",
    "нельзя сделать однозначный вывод",
]


def is_refusal(text: str) -> bool:

    return contains(
        text,
        REFUSAL_PATTERNS
    )


# =========================================================
# УТВЕРЖДЕНИЕ / ОТРИЦАНИЕ
# =========================================================

def contains_negation(text: str) -> bool:

    text = normalize(text)

    patterns = [
        "нет ",
        "не ",
        "нельзя",
        "невозможно",
        "отсутствует",
        "отсутствуют",
        "не следует",
        "не позволяет",
        "не указано",
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


# =========================================================
# SUPPORTED
# =========================================================

def evaluate_supported(
    answer: str,
    truth: dict
) -> str:

    expected = truth["expected_value"]

    expected_strings = [
        str(expected)
    ]

    if isinstance(expected, bool):

        if expected:
            expected_strings.extend([
                "включена",
                "включен",
                "включено",
                "используется",
            ])

        else:
            expected_strings.extend([
                "отключена",
                "отключен",
                "отключено",
            ])

    # Специальные варианты формулировок.
    if truth["source_path"].endswith(
        "retention_days"
    ):
        expected_strings.extend([
            "30 дней"
        ])

    if truth["source_path"].endswith(
        "frequency"
    ):
        expected_strings.extend([
            "ежедневно"
        ])

    if truth["source_path"].endswith(
        ".time"
    ):
        expected_strings.extend([
            "02:00",
            "2:00"
        ])

    if contains(answer, expected_strings):

        return "CORRECT"

    if is_refusal(answer):

        return "PARTIAL"

    return "HALLUCINATION"


# =========================================================
# UNSUPPORTED
# =========================================================

def evaluate_unsupported(
    answer: str,
    truth: dict
) -> str:

    # Если модель честно сообщает, что информации нет.
    if is_refusal(answer):

        return "CORRECT_REFUSAL"

    # Любой конкретный ответ на вопрос,
    # для которого в KB нет данных, является
    # неподтвержденным утверждением.
    return "HALLUCINATION"


# =========================================================
# CONTRADICTORY
# =========================================================

def evaluate_contradictory(
    answer: str,
    truth: dict
) -> str:

    wrong = truth.get(
        "false_premise"
    )

    correct = truth.get(
        "expected_value"
    )

    # -----------------------------------------------------
    # Сначала проверяем правильное исправление.
    # -----------------------------------------------------

    correct_values = [
        str(correct)
    ]

    if truth["source_path"].endswith(
        "frequency"
    ):
        correct_values.append("ежедневно")

    if truth["source_path"].endswith(
        "enabled"
    ):
        correct_values.extend([
            "включена",
            "включен",
            "включено",
        ])

    if truth["source_path"].endswith(
        "priority_levels"
    ):
        correct_values.extend([
            "4",
            "четыре",
            "четыре уровня",
            "4 уровня",
        ])

    if truth["source_path"].endswith(
        "head"
    ):
        correct_values.append(
            "Алексей Воронов"
        )

    if truth["source_path"].endswith(
        "database"
    ):
        correct_values.append(
            "PostgreSQL"
        )

    if contains(
        answer,
        correct_values
    ):

        return "CORRECT"

    # -----------------------------------------------------
    # Затем проверяем повторение ложной предпосылки.
    # -----------------------------------------------------

    if wrong and contains(
        answer,
        [wrong]
    ):

        return "CONTRADICTION"

    # -----------------------------------------------------
    # Модель заметила ошибочную предпосылку,
    # но не привела правильное значение.
    # -----------------------------------------------------

    if is_refusal(answer):

        return "PARTIAL"

    if contains(
        answer,
        [
            "неверно",
            "ошибочно",
            "это не так",
            "не соответствует",
            "на самом деле",
            "ошибочная предпосылка",
            "ложная предпосылка",
        ]
    ):

        return "PARTIAL"

    return "HALLUCINATION"


# =========================================================
# INFERENCE
# =========================================================

def evaluate_inference(
    answer: str,
    truth: dict
) -> str:

    invalid_inference = truth.get(
        "invalid_inference"
    )

    # -----------------------------------------------------
    # Если модель прямо повторила недопустимый вывод.
    # -----------------------------------------------------

    if invalid_inference:

        if contains(
            answer,
            [invalid_inference]
        ):

            # Если одновременно явно отрицает этот вывод,
            # считаем ответ корректным.
            if contains(
                answer,
                [
                    "нельзя",
                    "невозможно",
                    "неверно",
                    "не так",
                    "не следует",
                    "не позволяет",
                ]
            ):
                return "CORRECT_REFUSAL"

            return "HALLUCINATION"

    # -----------------------------------------------------
    # Правильная стратегия для inference-вопросов —
    # не делать вывод, которого нет в KB.
    # -----------------------------------------------------

    if is_refusal(answer):

        return "CORRECT_REFUSAL"

    if contains(
        answer,
        [
            "нельзя сделать вывод",
            "нельзя утверждать",
            "невозможно сделать вывод",
            "невозможно утверждать",
            "нельзя сказать",
            "невозможно сказать",
            "недостаточно информации",
            "недостаточно данных",
            "не позволяет сделать вывод",
            "не следует из",
        ]
    ):

        return "CORRECT_REFUSAL"

    return "HALLUCINATION"


# =========================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =========================================================

def evaluate_answer(
    question_id: str,
    answer: str
) -> str:

    truth = get_ground_truth(
        question_id
    )

    category = truth["category"]

    if category == "supported":

        return evaluate_supported(
            answer,
            truth
        )

    if category == "unsupported":

        return evaluate_unsupported(
            answer,
            truth
        )

    if category == "contradictory":

        return evaluate_contradictory(
            answer,
            truth
        )

    if category == "inference":

        return evaluate_inference(
            answer,
            truth
        )

    return "PARTIAL"


# =========================================================
# MAIN
# =========================================================

def main():

    raw_results = load_json(
        RAW_RESULTS_FILE
    )

    question_ids = [
        item["question_id"]
        for item in raw_results
    ]

    validate_ground_truth(
        list(GROUND_TRUTH.keys())
    )

    evaluated = []

    for item in raw_results:

        question_id = item["question_id"]
        answer = item.get(
            "answer",
            ""
        )

        classification = evaluate_answer(
            question_id,
            answer
        )

        evaluated.append({
            **item,
            "evaluation": {
                "classification": classification,
                "ground_truth": GROUND_TRUTH[
                    question_id
                ],
            }
        })

    # -----------------------------------------------------
    # Сохраняем
    # -----------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluated,
            file,
            ensure_ascii=False,
            indent=2
        )

    # -----------------------------------------------------
    # Статистика
    # -----------------------------------------------------

    statistics = Counter(
        (
            item["mode"],
            item["evaluation"]["classification"]
        )
        for item in evaluated
    )

    classifications = [
        "CORRECT",
        "CORRECT_REFUSAL",
        "HALLUCINATION",
        "CONTRADICTION",
        "PARTIAL",
    ]

    print()
    print("=" * 65)
    print("ОЦЕНКА ЭКСПЕРИМЕНТА")
    print("=" * 65)

    for mode in [
        "baseline",
        "prompt"
    ]:

        print()
        print(
            f"РЕЖИМ: {mode.upper()}"
        )

        total = 0

        for classification in classifications:

            count = statistics[
                (
                    mode,
                    classification
                )
            ]

            total += count

            print(
                f"{classification:20} {count}"
            )

        print(
            f"{'TOTAL':20} {total}"
        )

    # -----------------------------------------------------
    # Hallucination Rate
    # -----------------------------------------------------

    print()
    print("=" * 65)
    print("HALLUCINATION RATE")
    print("=" * 65)

    for mode in [
        "baseline",
        "prompt"
    ]:

        hallucinations = statistics[
            (
                mode,
                "HALLUCINATION"
            )
        ]

        total = sum(
            statistics[
                (
                    mode,
                    classification
                )
            ]
            for classification in classifications
        )

        rate = (
            hallucinations
            / total
            * 100
        )

        print(
            f"{mode.upper():10} "
            f"{hallucinations}/{total} "
            f"= {rate:.2f}%"
        )

    print()
    print(
        f"Результаты сохранены:"
    )
    print(OUTPUT_FILE)
    print()


if __name__ == "__main__":
    main()