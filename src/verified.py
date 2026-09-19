import json
import os
import re
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KB_FILE = BASE_DIR / "data" / "knowledge_base.json"

QUESTIONS_FILE = BASE_DIR / "data" / "questions.json"

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
    / "verified_experiment_v1.json"
)


# =========================================================
# API
# =========================================================

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Не найден DEEPSEEK_API_KEY в файле .env"
    )


client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-chat"


# =========================================================
# НАСТРОЙКИ
# =========================================================

TEMPERATURE = 0

# Пауза между запросами.
REQUEST_DELAY = 1.0


# =========================================================
# ЗАГРУЗКА JSON
# =========================================================

def load_json(path: Path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_results_json(path: Path):
    """
    Безопасная загрузка файла результатов.

    Если файл отсутствует, пустой или поврежден,
    начинаем эксперимент заново.
    """

    if not path.exists():
        return []

    if path.stat().st_size == 0:
        print(
            "Предупреждение: файл результатов пустой. "
            "Начинаем эксперимент заново."
        )
        return []

    try:
        data = load_json(path)

        if isinstance(data, list):
            return data

        print(
            "Предупреждение: файл результатов "
            "содержит не список. Начинаем заново."
        )

        return []

    except json.JSONDecodeError:
        print(
            "Предупреждение: файл результатов поврежден "
            "или содержит некорректный JSON. "
            "Начинаем эксперимент заново."
        )

        return []


def load_baseline_answers():
    """
    Загружает готовые BASELINE-ответы
    из raw_results.json.

    VERIFIED не генерирует новый исходный ответ.
    Он проверяет именно тот ответ, который
    уже был получен в BASELINE.
    """

    if not RAW_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Не найден файл BASELINE-результатов:\n"
            f"{RAW_RESULTS_FILE}"
        )

    raw_results = load_json(RAW_RESULTS_FILE)

    baseline = {}

    for item in raw_results:

        if item.get("mode") != "baseline":
            continue

        question_id = item.get("question_id")
        answer = item.get("answer")

        if question_id and answer:
            baseline[question_id] = answer

    return baseline


def save_json(path: Path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


# =========================================================
# ЗАГРУЗКА ДАННЫХ
# =========================================================

knowledge_base = load_json(
    KB_FILE
)

questions = load_json(
    QUESTIONS_FILE
)

baseline_answers = load_baseline_answers()


# =========================================================
# KB → ТЕКСТ
# =========================================================

def knowledge_base_to_text():

    return json.dumps(
        knowledge_base,
        ensure_ascii=False,
        indent=2
    )


KB_TEXT = knowledge_base_to_text()


# =========================================================
# НОРМАЛИЗАЦИЯ
# =========================================================

def normalize(text: str) -> str:

    text = str(text).lower()

    text = text.replace(
        "ё",
        "е"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# JSON ИЗ ОТВЕТА LLM
# =========================================================

def extract_json(text: str):

    if not text:
        return None

    text = text.strip()

    # -----------------------------------------------------
    # 1. Обычный JSON
    # -----------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # -----------------------------------------------------
    # 2. Markdown JSON
    # -----------------------------------------------------

    cleaned = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace(
        "```",
        ""
    ).strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    # -----------------------------------------------------
    # 3. Ищем объект
    # -----------------------------------------------------

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:

        candidate = cleaned[
            start:end + 1
        ]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    # -----------------------------------------------------
    # 4. Ищем массив
    # -----------------------------------------------------

    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start != -1 and end != -1:

        candidate = cleaned[
            start:end + 1
        ]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    return None


# =========================================================
# API REQUEST
# =========================================================

def chat(
    system_prompt: str,
    user_prompt: str
):

    response = client.chat.completions.create(

        model=MODEL,

        temperature=TEMPERATURE,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# =========================================================
# ШАГ 1.
# ВЫДЕЛЕНИЕ УТВЕРЖДЕНИЙ
# =========================================================


def extract_claims(
    question: str,
    answer: str
) -> list[str]:

    system_prompt = """
Ты — модуль выделения ФАКТИЧЕСКИХ УТВЕРЖДЕНИЙ
из ответа информационной системы.

Твоя задача — извлечь ВСЕ существенные утверждения
о предметной области из ответа.

ОЧЕНЬ ВАЖНО:

Ты НЕ должен решать, является утверждение
правильным или неправильным.

Твоя задача только извлечь утверждения.

Даже если утверждение кажется тебе очевидно
ложным, противоречащим базе знаний или слишком
сильным — ВСЁ РАВНО извлеки его.

Именно следующий модуль будет решать,
SUPPORTED это, UNSUPPORTED или CONTRADICTED.

--------------------------------------------------
ЧТО НУЖНО ИЗВЛЕКАТЬ
--------------------------------------------------

Извлекай:

- основные ответы на вопрос;
- утверждения о фактах;
- утверждения о свойствах объектов;
- утверждения о связях между объектами;
- причинно-следственные утверждения;
- утверждения с количественными значениями;
- утверждения с ограничителями области действия.

Особенно внимательно сохраняй слова:

«все»
«каждый»
«только»
«всегда»
«никто»
«обязательно»
«точно»
«именно»

Они могут принципиально менять смысл утверждения.

--------------------------------------------------
ПРИМЕР 1
--------------------------------------------------

Вопрос:

«Можно ли утверждать, что все сотрудники используют
Active Directory для аутентификации?»

Ответ:

«Да, можно утверждать, что все сотрудники используют
Active Directory для аутентификации — в базе знаний
указано, что аутентификация осуществляется через
Active Directory.»

Нужно извлечь:

[
  "Все сотрудники используют Active Directory
   для аутентификации.",
  "Аутентификация в организации осуществляется
   через Active Directory."
]

Первое утверждение ОБЯЗАТЕЛЬНО должно быть извлечено,
даже если оно впоследствии окажется UNSUPPORTED.

--------------------------------------------------
ПРИМЕР 2
--------------------------------------------------

Вопрос:

«Можно ли утверждать, что корпоративный чат работает
через обычный HTTP-запрос?»

Ответ:

«Нет, нельзя. Корпоративный чат работает через WebSocket.»

Нужно извлечь:

[
  "Корпоративный чат работает через WebSocket."
]

Не нужно извлекать:

«Нельзя утверждать...»

потому что это не предметный факт.

--------------------------------------------------
ПРИМЕР 3
--------------------------------------------------

Ответ:

«В компании используется PostgreSQL, поэтому MySQL
не используется.»

Нужно извлечь:

[
  "В компании используется PostgreSQL.",
  "В компании не используется MySQL."
]

Оба утверждения нужно передать verifier.

--------------------------------------------------
МЕТА-УТВЕРЖДЕНИЯ
--------------------------------------------------

Не извлекай отдельными claims фразы,
которые описывают саму базу знаний:

«В базе знаний указано...»

«В базе знаний нет информации...»

«База знаний не позволяет утверждать...»

«На основании имеющихся данных нельзя сказать...»

Однако если внутри такой фразы содержится
предметное утверждение, извлеки само предметное
утверждение.

Например:

«В базе знаний указано, что все сотрудники
используют Active Directory.»

нужно преобразовать в:

«Все сотрудники используют Active Directory.»

--------------------------------------------------
ОТКАЗ ОТ ОТВЕТА
--------------------------------------------------

Если модель просто сообщает:

«В базе знаний нет информации о генеральном директоре.»

не нужно создавать claim:

«В базе знаний нет информации о генеральном директоре.»

Это мета-информация.

--------------------------------------------------
ЛОГИЧЕСКИЕ ВЫВОДЫ
--------------------------------------------------

Если модель делает вывод, извлеки сам вывод.

Например:

«WebSocket используется в системе, поэтому
корпоративный чат работает через WebSocket.»

Нужно извлечь:

[
  "WebSocket используется в системе.",
  "Корпоративный чат работает через WebSocket."
]

Второй claim должен быть передан verifier,
даже если он не следует из KB.

--------------------------------------------------
ВАЖНО
--------------------------------------------------

Не исправляй утверждения.

Не ослабляй их.

Не усиливай их.

Не проверяй их истинность.

Извлекай именно то, что утверждает модель.

Если модель сказала:

«Все сотрудники используют AD»

нужно вернуть:

«Все сотрудники используют AD.»

а НЕ:

«В организации используется AD.»

Это принципиально важно для эксперимента.

--------------------------------------------------
ФОРМАТ
--------------------------------------------------

Верни ТОЛЬКО JSON-массив строк:

[
  "утверждение 1",
  "утверждение 2"
]

Если в ответе вообще нет проверяемых
предметных утверждений:

[]
""".strip()

    user_prompt = f"""
БАЗА ЗНАНИЙ:

{KB_TEXT}

ВОПРОС:

{question}

ОТВЕТ МОДЕЛИ:

{answer}

Извлеки ВСЕ существенные предметные
утверждения из ответа.
Не проверяй их истинность.
""".strip()

    raw = chat(
        system_prompt,
        user_prompt
    )

    try:

        data = extract_json(raw)

        if not isinstance(data, list):
            return []

        claims = []

        for item in data:

            if not isinstance(item, str):
                continue

            claim = item.strip()

            if not claim:
                continue

            if claim not in claims:
                claims.append(claim)

        return claims

    except Exception as error:

        print(
            f"Ошибка извлечения claims: {error}"
        )

        return []

# =========================================================
# ШАГ 2.
# ПРОВЕРКА CLAIMS ПО KB
# =========================================================

def verify_claims(
    question: str,
    claims: list[str]
) -> list[dict]:

    if not claims:
        return []

    claims_text = "\n".join(
        f"{index + 1}. {claim}"
        for index, claim in enumerate(claims)
    )

    system_prompt = """
Ты — строгий модуль семантической проверки
фактических утверждений.

Проверяй каждое утверждение ТОЛЬКО
относительно предоставленной базы знаний.

Для каждого утверждения используй ровно
один из трех статусов:

SUPPORTED
UNSUPPORTED
CONTRADICTED

--------------------------------------------------
SUPPORTED
--------------------------------------------------

Используй SUPPORTED только тогда, когда
база знаний непосредственно подтверждает
утверждение целиком.

Должны совпадать:

- объект;
- свойство;
- значение;
- необходимые отношения;
- область действия утверждения.

Наличие отдельных совпадающих слов
недостаточно.

--------------------------------------------------
UNSUPPORTED
--------------------------------------------------

Используй UNSUPPORTED, если база знаний
не содержит достаточной информации
для подтверждения или опровержения
утверждения.

--------------------------------------------------
CONTRADICTED
--------------------------------------------------

Используй CONTRADICTED, если база знаний
содержит информацию, непосредственно
противоречащую утверждению.

--------------------------------------------------
КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА
--------------------------------------------------

1. НЕ ДЕЛАЙ ЛОГИЧЕСКИХ ВЫВОДОВ,
КОТОРЫХ НЕТ В KB.

KB:

authentication = Active Directory

Подтверждается:

«В организации используется Active Directory
для аутентификации.»

НЕ подтверждается:

«Все сотрудники используют Active Directory.»

Статус:

UNSUPPORTED

--------------------------------------------------

2. НЕ ПЕРЕНОСИ СВОЙСТВА ОДНОГО ОБЪЕКТА
НА ДРУГОЙ.

KB:

websocket = true

Подтверждается:

«В системе используется WebSocket.»

НЕ подтверждается:

«Корпоративный чат работает через WebSocket.»

если связь между WebSocket и корпоративным
чатом непосредственно не указана.

Статус:

UNSUPPORTED

--------------------------------------------------

3. НЕ РАСШИРЯЙ ОБЛАСТЬ ДЕЙСТВИЯ.

KB:

MFA применяется к административным
учетным записям.

НЕ следует из этого:

«MFA применяется ко всем сотрудникам.»

Статус:

UNSUPPORTED

--------------------------------------------------

4. ПРОТИВОРЕЧИЕ.

KB:

database = PostgreSQL

Утверждение:

«В системе используется MySQL.»

Статус:

CONTRADICTED

--------------------------------------------------

5. ЧИСЛОВЫЕ ЗНАЧЕНИЯ.

KB:

retention_days = 30

«хранятся 30 дней»
→ SUPPORTED

«хранятся более 30 дней»
→ CONTRADICTED

«хранятся менее 30 дней»
→ CONTRADICTED

«срок хранения неизвестен»
→ CONTRADICTED

--------------------------------------------------

6. НЕ ПУТАЙ ОТСУТСТВИЕ ФАКТА
С ПРОТИВОРЕЧИЕМ.

Если KB ничего не говорит о директоре,
то:

«Директор — Иван Петров»

UNSUPPORTED,

а не CONTRADICTED.

--------------------------------------------------

7. НЕ ПУТАЙ ОБЩИЙ ФАКТ
С БОЛЕЕ СИЛЬНЫМ УТВЕРЖДЕНИЕМ.

Например:

KB:

«Аутентификация — Active Directory.»

Утверждение:

«Все сотрудники проходят аутентификацию
через Active Directory.»

UNSUPPORTED.

--------------------------------------------------

8. Не считай отрицательное утверждение автоматически
противоречащим базе знаний.

Например, если KB содержит:

database = PostgreSQL

то:

«В системе используется PostgreSQL»
→ SUPPORTED

Но:

«В системе не используется MySQL»
→ UNSUPPORTED,

если в KB отсутствует прямое утверждение
об отсутствии MySQL.

А:

«В системе используется MySQL»
→ CONTRADICTED,

поскольку это непосредственно противоречит
указанному значению PostgreSQL.

Верни ТОЛЬКО JSON-массив:

[
  {
    "claim": "точный текст утверждения",
    "status": "SUPPORTED",
    "reason": "краткое объяснение",
    "evidence": [
      "конкретный факт из базы знаний"
    ]
  }
]

Для каждого claim должен быть ровно
один статус:

SUPPORTED
UNSUPPORTED
CONTRADICTED
""".strip()

    user_prompt = f"""
ВОПРОС:

{question}

БАЗА ЗНАНИЙ:

{KB_TEXT}

УТВЕРЖДЕНИЯ:

{claims_text}

Проверь каждое утверждение.
""".strip()

    raw = chat(
        system_prompt,
        user_prompt
    )

    try:

        data = extract_json(raw)

        if not isinstance(data, list):
            return []

        verified = []

        for item in data:

            if not isinstance(item, dict):
                continue

            claim = item.get("claim", "")
            status = item.get("status", "")
            reason = item.get("reason", "")
            evidence = item.get("evidence", [])

            status = str(status).upper().strip()

            if status not in {
                "SUPPORTED",
                "UNSUPPORTED",
                "CONTRADICTED"
            }:
                continue

            if not isinstance(evidence, list):
                evidence = []

            verified.append({
                "claim": str(claim).strip(),
                "status": status,
                "reason": str(reason).strip(),
                "evidence": [
                    str(value).strip()
                    for value in evidence
                ]
            })

        return verified

    except Exception as error:

        print(
            f"Ошибка проверки claims: {error}"
        )

        return []


# =========================================================
# ШАГ 3.
# ФОРМИРОВАНИЕ ИСПРАВЛЕННОГО ОТВЕТА
# =========================================================

def build_final_answer(
    question: str,
    initial_answer: str,
    verified_claims: list[dict]
):

    verification_text = json.dumps(
        verified_claims,
        ensure_ascii=False,
        indent=2
    )

    system_prompt = """
Ты формируешь финальный ответ пользователю
после автоматической проверки утверждений.

Используй ТОЛЬКО информацию,
подтвержденную базой знаний.

ПРАВИЛА:

1. SUPPORTED можно использовать.

2. UNSUPPORTED нельзя выдавать как установленный факт.

3. CONTRADICTED нельзя повторять
как правильное утверждение.

4. Если CONTRADICTED-утверждение имеет
фактическую альтернативу в KB,
укажи правильное значение.

5. Если информации недостаточно,
прямо скажи об этом.

6. Не придумывай:
- причины;
- даты;
- числа;
- имена;
- связи между объектами;
- дополнительные характеристики.

7. Не упоминай:
- verifier;
- claims;
- JSON;
- внутреннюю проверку;
- этапы алгоритма.

8. Если исходный ответ сделал вывод,
который сильнее, чем позволяет KB,
убери этот вывод.

9. Особенно внимательно относись
к словам:

«все»
«каждый»
«всегда»
«только»
«обязательно»
«точно»
«именно»

Такие слова нельзя использовать,
если соответствующая область действия
не подтверждается непосредственно KB.

10. Если вопрос содержит ложную предпосылку,
не принимай ее. Исправь предпосылку
и сообщи фактическое состояние,
если оно есть в KB.

Ответ должен быть естественным,
кратким и понятным.
""".strip()

    user_prompt = f"""
БАЗА ЗНАНИЙ:

{KB_TEXT}

ВОПРОС:

{question}

ИСХОДНЫЙ ОТВЕТ:

{initial_answer}

РЕЗУЛЬТАТ ПРОВЕРКИ:

{verification_text}

Сформируй финальный ответ.
""".strip()

    return chat(
        system_prompt,
        user_prompt
    )


# =========================================================
# ШАГ 4.
# ПОВТОРНАЯ ПРОВЕРКА ФИНАЛЬНОГО ОТВЕТА
# =========================================================

def final_verification(
    question: str,
    answer: str
):

    final_claims = extract_claims(
        question,
        answer
    )

    final_verified = verify_claims(
        question,
        final_claims
    )

    return (
        final_claims,
        final_verified
    )


# =========================================================
# ОДИН ВОПРОС
# =========================================================

def process_question(
    question_item: dict,
    baseline_answers: dict
):

    question_id = question_item["id"]

    category = question_item["category"]

    question = question_item["question"]

    print()
    print("-" * 60)

    print(
        f"{question_id} | {category}"
    )

    print(question)

    started = time.time()

    # -----------------------------------------------------
    # 1. BASELINE answer
    # -----------------------------------------------------

    initial_answer = baseline_answers.get(
        question_id
    )

    if not initial_answer:

        raise ValueError(
            f"Не найден BASELINE-ответ "
            f"для {question_id}"
        )

    print()
    print("INITIAL:")
    print(initial_answer)

    time.sleep(
        REQUEST_DELAY
    )

    # -----------------------------------------------------
    # 2. Claims
    # -----------------------------------------------------

    claims = extract_claims(
        question,
        initial_answer
    )

    print()
    print(
        f"CLAIMS: {len(claims)}"
    )

    for claim in claims:

        print(
            f"  - {claim}"
        )

    time.sleep(
        REQUEST_DELAY
    )

    # -----------------------------------------------------
    # 3. Verification
    # -----------------------------------------------------

    verified_claims = verify_claims(
        question,
        claims
    )

    print()
    print("VERIFICATION:")

    for item in verified_claims:

        print(
            f"  [{item['status']}] "
            f"{item['claim']}"
        )

    # -----------------------------------------------------
    # 4. Нужно ли исправлять ответ?
    # -----------------------------------------------------

    needs_revision = any(
        item["status"] in {
            "UNSUPPORTED",
            "CONTRADICTED"
        }
        for item in verified_claims
    )

    # -----------------------------------------------------
    # 5. Final answer
    # -----------------------------------------------------

    if needs_revision:

        time.sleep(
            REQUEST_DELAY
        )

        final_answer = build_final_answer(
            question,
            initial_answer,
            verified_claims
        )

    else:

        final_answer = initial_answer

    # -----------------------------------------------------
    # 6. Final verification
    # -----------------------------------------------------

    time.sleep(
        REQUEST_DELAY
    )

    final_claims, final_verified = (
        final_verification(
            question,
            final_answer
        )
    )

    elapsed = time.time() - started

    # -----------------------------------------------------
    # PRINT RESULT
    # -----------------------------------------------------

    print()
    print("FINAL:")
    print(final_answer)

    print()
    print("FINAL VERIFICATION:")

    for item in final_verified:

        print(
            f"  [{item['status']}] "
            f"{item['claim']}"
        )

    print()
    print(
        f"TIME: {elapsed:.2f} sec"
    )

    # -----------------------------------------------------
    # RESULT OBJECT
    # -----------------------------------------------------

    return {

        "question_id": question_id,

        "category": category,

        "question": question,

        "mode": "verified",

        # Это именно BASELINE-ответ.
        # VERIFIED выполняет его постгенерационную
        # проверку и исправление.
        "initial_answer": initial_answer,

        "initial_claims": claims,

        "initial_verification": verified_claims,

        "revision_required": needs_revision,

        "final_answer": final_answer,

        "final_claims": final_claims,

        "final_verification": final_verified,

        "response_time_seconds": round(
            elapsed,
            3
        ),

        "timestamp": datetime.now().isoformat(),

        "error": None,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("VERIFIED EXPERIMENT")
    print("=" * 60)

    print(
        f"Вопросов: {len(questions)}"
    )

    print(
        f"Модель: {MODEL}"
    )

    print(
        f"Результаты: {OUTPUT_FILE}"
    )

    # -----------------------------------------------------
    # Загружаем предыдущие результаты.
    # -----------------------------------------------------

    results = load_results_json(
        OUTPUT_FILE
    )

    completed_ids = {
        item["question_id"]
        for item in results
        if item.get("error") is None
    }

    print()
    print(
        f"Уже выполнено: "
        f"{len(completed_ids)}"
    )

    # -----------------------------------------------------
    # ТЕСТОВЫЙ ЗАПУСК
    #
    # После успешной проверки четырех вопросов
    # заменим этот список на:
    #
    # test_questions = questions
    #
    # -----------------------------------------------------

    test_questions = questions

    total = len(test_questions)

    # -----------------------------------------------------
    # Обработка
    # -----------------------------------------------------

    for index, question in enumerate(
        test_questions,
        start=1
    ):

        question_id = question["id"]

        if question_id in completed_ids:

            print()
            print(
                f"[{index}/{total}] "
                f"{question_id} — пропуск"
            )

            continue

        print()
        print(
            f"[{index}/{total}]"
        )

        try:

            result = process_question(
                question,
                baseline_answers
            )

            results.append(
                result
            )

            save_json(
                OUTPUT_FILE,
                results
            )

            print()
            print(
                "Результат сохранён."
            )

            completed_ids.add(
                question_id
            )

        except Exception as error:

            print()
            print(
                f"ОШИБКА: {error}"
            )

            error_result = {

                "question_id": question_id,

                "category": question["category"],

                "question": question["question"],

                "mode": "verified",

                "initial_answer": "",

                "initial_claims": [],

                "initial_verification": [],

                "revision_required": False,

                "final_answer": "",

                "final_claims": [],

                "final_verification": [],

                "response_time_seconds": 0,

                "timestamp": datetime.now().isoformat(),

                "error": str(error),
            }

            results.append(
                error_result
            )

            save_json(
                OUTPUT_FILE,
                results
            )

        time.sleep(
            REQUEST_DELAY
        )

    # -----------------------------------------------------
    # Итог
    # -----------------------------------------------------

    successful = [
        item
        for item in results
        if item.get("error") is None
    ]

    errors = [
        item
        for item in results
        if item.get("error")
    ]

    print()
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 60)

    print(
        f"Успешно: {len(successful)}"
    )

    print(
        f"Ошибок: {len(errors)}"
    )

    print()
    print("Результаты:")

    print(
        OUTPUT_FILE
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()