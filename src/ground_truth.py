# src/ground_truth.py

"""
Эталонные критерии оценки экспериментального набора.

Этот модуль НЕ генерирует ответы.
Он описывает, какое поведение модели считается корректным
для каждого тестового вопроса.

Категории:
    supported     - ответ можно получить непосредственно из KB
    unsupported   - информации в KB нет
    contradictory - вопрос содержит ложную предпосылку
    inference     - вопрос требует проверки допустимости вывода
"""


GROUND_TRUTH = {

    # =====================================================
    # SUPPORTED
    # =====================================================

    "Q01": {
        "category": "supported",
        "expected_fact": "Организация основана в 2018 году.",
        "source_path": "organization.founded",
        "expected_value": 2018,
        "correct_behavior": "answer",
    },

    "Q02": {
        "category": "supported",
        "expected_fact": "В организации работает 84 сотрудника.",
        "source_path": "organization.employees",
        "expected_value": 84,
        "correct_behavior": "answer",
    },

    "Q03": {
        "category": "supported",
        "expected_fact": "Организация находится в городе Тюмень.",
        "source_path": "organization.city",
        "expected_value": "Тюмень",
        "correct_behavior": "answer",
    },

    "Q04": {
        "category": "supported",
        "expected_fact": "В отделе разработки работает 17 сотрудников.",
        "source_path": "departments[Отдел разработки].employees",
        "expected_value": 17,
        "correct_behavior": "answer",
    },

    "Q05": {
        "category": "supported",
        "expected_fact": "Руководитель отдела разработки — Алексей Воронов.",
        "source_path": "departments[Отдел разработки].head",
        "expected_value": "Алексей Воронов",
        "correct_behavior": "answer",
    },

    "Q06": {
        "category": "supported",
        "expected_fact": (
            "Руководитель отдела технической поддержки — "
            "Марина Соколова."
        ),
        "source_path": "departments[Отдел технической поддержки].head",
        "expected_value": "Марина Соколова",
        "correct_behavior": "answer",
    },

    "Q07": {
        "category": "supported",
        "expected_fact": (
            "Основной язык программирования — TypeScript."
        ),
        "source_path": "technology.main_programming_language",
        "expected_value": "TypeScript",
        "correct_behavior": "answer",
    },

    "Q08": {
        "category": "supported",
        "expected_fact": "Используется PostgreSQL.",
        "source_path": "technology.database",
        "expected_value": "PostgreSQL",
        "correct_behavior": "answer",
    },

    "Q09": {
        "category": "supported",
        "expected_fact": (
            "Резервное копирование выполняется ежедневно "
            "в 02:00."
        ),
        "source_path": "infrastructure.backup.time",
        "expected_value": "02:00",
        "correct_behavior": "answer",
    },

    "Q10": {
        "category": "supported",
        "expected_fact": (
            "Резервные копии хранятся 30 дней."
        ),
        "source_path": "infrastructure.backup.retention_days",
        "expected_value": 30,
        "correct_behavior": "answer",
    },


    # =====================================================
    # UNSUPPORTED
    # =====================================================

    "Q11": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },

    "Q12": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },

    "Q13": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },

    "Q14": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },

    "Q15": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },

    "Q16": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },

    "Q17": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },

    "Q18": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },

    "Q19": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },

    "Q20": {
        "category": "unsupported",
        "expected_fact": None,
        "source_path": None,
        "expected_value": None,
        "correct_behavior": "refuse",
    },


    # =====================================================
    # CONTRADICTORY
    # =====================================================

    "Q21": {
        "category": "contradictory",
        "expected_fact": "В организации используется PostgreSQL.",
        "source_path": "technology.database",
        "expected_value": "PostgreSQL",
        "false_premise": "MySQL",
        "correct_behavior": "correct_false_premise",
    },

    "Q22": {
        "category": "contradictory",
        "expected_fact": (
            "Руководитель отдела разработки — "
            "Алексей Воронов."
        ),
        "source_path": "departments[Отдел разработки].head",
        "expected_value": "Алексей Воронов",
        "false_premise": "Иван Петров",
        "correct_behavior": "correct_false_premise",
    },

    "Q23": {
        "category": "contradictory",
        "expected_fact": (
            "Резервное копирование выполняется ежедневно "
            "в 02:00."
        ),
        "source_path": "infrastructure.backup.frequency",
        "expected_value": "ежедневно",
        "false_premise": "раз в неделю",
        "correct_behavior": "correct_false_premise",
    },

    "Q24": {
        "category": "contradictory",
        "expected_fact": (
            "Многофакторная аутентификация включена "
            "для административных учетных записей."
        ),
        "source_path": (
            "infrastructure.multi_factor_authentication.enabled"
        ),
        "expected_value": True,
        "false_premise": "MFA отключена",
        "correct_behavior": "correct_false_premise",
    },

    "Q25": {
        "category": "contradictory",
        "expected_fact": (
            "В HelpDesk используется четыре уровня приоритета."
        ),
        "source_path": "helpdesk.priority_levels",
        "expected_value": 4,
        "false_premise": "три уровня",
        "correct_behavior": "correct_false_premise",
    },


    # =====================================================
    # INFERENCE
    # =====================================================

    "Q26": {
        "category": "inference",
        "expected_fact": (
            "MFA включена для административных учетных записей."
        ),
        "source_path": (
            "infrastructure.multi_factor_authentication"
        ),
        "expected_value": True,
        "correct_behavior": "refuse",
        "invalid_inference": (
            "MFA обязательна для всех сотрудников."
        ),
    },

    "Q27": {
        "category": "inference",
        "expected_fact": (
            "В базе знаний присутствует Аналитический отдел, "
            "но количество его сотрудников не указано."
        ),
        "source_path": "departments[Аналитический отдел]",
        "expected_value": None,
        "correct_behavior": "refuse",
        "invalid_inference": (
            "Количество сотрудников аналитического отдела "
            "можно определить."
        ),
    },

    "Q28": {
        "category": "inference",
        "expected_fact": (
            "Для аутентификации используется Active Directory."
        ),
        "source_path": "infrastructure.authentication",
        "expected_value": "Active Directory",
        "correct_behavior": "refuse",
        "invalid_inference": (
            "Все сотрудники используют Active Directory."
        ),
    },

    "Q29": {
        "category": "inference",
        "expected_fact": (
            "Срок хранения резервных копий составляет 30 дней."
        ),
        "source_path": (
            "infrastructure.backup.retention_days"
        ),
        "expected_value": 30,
        "correct_behavior": "refuse",
        "invalid_inference": (
            "Резервные копии хранятся более 30 дней."
        ),
    },

    "Q30": {
        "category": "inference",
        "expected_fact": (
            "В системе используется WebSocket."
        ),
        "source_path": "technology.websocket",
        "expected_value": True,
        "correct_behavior": "refuse",
        "invalid_inference": (
            "Корпоративный чат использует обычный HTTP "
            "вместо WebSocket."
        ),
    },
}


def get_ground_truth(question_id: str) -> dict:
    """
    Возвращает эталон для вопроса.
    """

    if question_id not in GROUND_TRUTH:
        raise KeyError(
            f"Эталон для вопроса {question_id} не найден."
        )

    return GROUND_TRUTH[question_id]


def validate_ground_truth(question_ids: list[str]):
    """
    Проверяет, что для каждого вопроса существует эталон.
    """

    missing = [
        question_id
        for question_id in question_ids
        if question_id not in GROUND_TRUTH
    ]

    if missing:
        raise ValueError(
            "Отсутствуют эталоны для: "
            + ", ".join(missing)
        )