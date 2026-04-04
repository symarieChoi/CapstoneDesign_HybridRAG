import os

SOURCE_TO_NOTICE_IDS = {
    "▣ 글솝 졸업요건 기술창업역량(현장실습, 창업교과목, 스타트업창.txt": [
        "NOTICE_GLSW_STARTUP_001",
        "NOTICE_GLSW_STARTUP_002",
        "NOTICE_GLSW_STARTUP_003",
        "NOTICE_GLSW_STARTUP_004"
    ],
    "글로벌소프트웨어융합전공 소속 학생이 이수한 교과목의 교과구분.txt": [
        "NOTICE_GLSW_CLASSIFICATION_001",
        "NOTICE_GLSW_CLASSIFICATION_002"
    ],
    "컴퓨터학부(글로벌소프트웨어융합전공).txt": [
        "NOTICE_GLSW_FAQ_001",
        "NOTICE_GLSW_FAQ_002",
        "NOTICE_GLSW_FAQ_003",
        "NOTICE_GLSW_FAQ_004",
        "NOTICE_GLSW_FAQ_005"
    ],
    "붙임_교양초과이수자_교과구분변경_관련_매뉴얼-학생용.pdf": [
        "NOTICE_GLSW_MANUAL_001"
    ],
    "졸업 사정 기준2025-글로벌소프트웨어융합전공-2p.pdf": [
        "NOTICE_GLSW_GRADSTD_001",
        "NOTICE_GLSW_GRADSTD_002",
        "NOTICE_GLSW_GRADSTD_003",
        "NOTICE_GLSW_GRADSTD_004"
    ],
    "별첨 도전 K-스타트업 2025 부처 통합 창업경진대회 공고최종.pdf": [
        "NOTICE_STARTUP_CONTEST_001",
        "NOTICE_STARTUP_CONTEST_002",
        "NOTICE_STARTUP_CONTEST_003"
    ],
    "참고자료4 창업대체기준창업경진대회 목록 포함2025.pdf": [
        "NOTICE_GLSW_STARTUP_005"
    ],
    "학생작성_교양초과이수_교과구분_변경_신청서-홍길동-제출일자.pdf": [
        "NOTICE_GLSW_FORM_001"
    ],
    "참고자료 창업교과목_인정목록_2026.xlsx": [
        "NOTICE_GLSW_ENTRE_COURSE_001",
        "NOTICE_GLSW_ENTRE_COURSE_002"
    ],
    "학생작성_교양초과이수_교과구분_변경_신청_내역-홍길동-제출일.xlsx": [
        "NOTICE_GLSW_FORM_002"
    ]
}

def source_to_ids(source_path: str):
    filename = os.path.basename(source_path)
    return SOURCE_TO_NOTICE_IDS.get(filename, [])