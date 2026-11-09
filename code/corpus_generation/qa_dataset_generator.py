"""
대규모 한국어 QA 데이터셋 생성 시스템
목표: 50만 건의 포멀한 전문 도메인 QA 데이터셋
"""

import os
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import aiohttp
from tqdm import tqdm

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
API_URL = "https://api.openai.com/v1/chat/completions"

# 도메인별 세부 키워드 정의
DOMAIN_KEYWORDS = {
    "행정_법규": {
        "subcategories": [
            "행정법", "민법", "형법", "상법", "노동법", "조세법", 
            "환경법", "건축법", "의료법", "교육법", "정보통신법", "금융법",
            "행정절차", "공공기관", "지방자치", "정부조직", "규제개혁", "법령해석"
        ],
        "question_types": ["법적 근거", "절차 안내", "요건 설명", "판례 분석", "법령 해석", "규제 내용"]
    },
    "경제_금융": {
        "subcategories": [
            "거시경제", "미시경제", "국제경제", "금융시장", "은행업", "증권",
            "보험", "외환", "파생상품", "자산관리", "기업금융", "재정정책",
            "통화정책", "무역", "투자", "회계", "세무", "경제지표"
        ],
        "question_types": ["개념 설명", "지표 분석", "정책 영향", "시장 메커니즘", "계산 방법", "제도 소개"]
    },
    "통계_데이터": {
        "subcategories": [
            "기술통계", "추론통계", "확률론", "회귀분석", "시계열분석", "베이즈통계",
            "다변량분석", "실험설계", "표본추출", "데이터마이닝", "빅데이터분석", "통계소프트웨어",
            "인구통계", "경제통계", "사회조사", "품질관리", "생물통계", "데이터시각화"
        ],
        "question_types": ["통계기법", "분석방법", "해석방법", "검정절차", "모형선택", "데이터처리"]
    },
    "공학_제조": {
        "subcategories": [
            "기계공학", "전기전자", "화학공학", "재료공학", "산업공학", "토목공학",
            "건축공학", "조선해양", "항공우주", "자동차", "로봇공학", "반도체",
            "생산관리", "품질관리", "공정설계", "설비관리", "안전공학", "스마트제조"
        ],
        "question_types": ["원리 설명", "설계 방법", "공정 절차", "재료 특성", "성능 평가", "문제 해결"]
    },
    "전산_알고리즘": {
        "subcategories": [
            "자료구조", "알고리즘설계", "계산복잡도", "정렬알고리즘", "탐색알고리즘", "그래프알고리즘",
            "동적프로그래밍", "분할정복", "탐욕알고리즘", "백트래킹", "암호학", "데이터베이스",
            "운영체제", "컴퓨터구조", "네트워크", "컴파일러", "인공지능", "병렬처리"
        ],
        "question_types": ["알고리즘 분석", "구현 방법", "시간복잡도", "최적화", "비교 분석", "응용 사례"]
    },
    "과학일반": {
        "subcategories": [
            "고전역학", "전자기학", "열역학", "양자역학", "상대성이론", "광학",
            "무기화학", "유기화학", "분석화학", "물리화학", "생화학", "고분자화학",
            "지구물리", "지질학", "해양학", "기상학", "천문학", "지구환경"
        ],
        "question_types": ["이론 설명", "법칙 유도", "실험 방법", "현상 해석", "계산 문제", "응용 분야"]
    },
    "환경_에너지": {
        "subcategories": [
            "대기환경", "수질환경", "토양환경", "폐기물", "환경영향평가", "기후변화",
            "신재생에너지", "태양광", "풍력", "수소에너지", "원자력", "에너지효율",
            "탄소중립", "환경정책", "환경모니터링", "생태복원", "환경경제", "지속가능성"
        ],
        "question_types": ["환경기준", "측정방법", "저감기술", "정책내용", "영향분석", "사례연구"]
    },
    "보건_공중보건": {
        "subcategories": [
            "역학", "감염병관리", "건강증진", "질병예방", "환경보건", "산업보건",
            "식품안전", "보건통계", "의료정책", "건강보험", "의료윤리", "보건행정",
            "정신건강", "모자보건", "노인보건", "의료정보", "보건경제", "국제보건"
        ],
        "question_types": ["질병특성", "예방방법", "관리체계", "정책제도", "통계해석", "연구방법"]
    },
    "교육_학술": {
        "subcategories": [
            "교육과정", "교수법", "교육평가", "교육심리", "교육철학", "교육사회학",
            "교육행정", "교육정책", "특수교육", "평생교육", "직업교육", "교육공학",
            "학술연구방법", "논문작성", "학술윤리", "학술정보", "연구윤리", "학위제도"
        ],
        "question_types": ["이론 설명", "방법론", "제도 안내", "평가 기준", "사례 분석", "개선 방안"]
    },
    "국방_안보": {
        "subcategories": [
            "국가안보전략", "국방정책", "군사전략", "국방예산", "병역제도", "군사교육",
            "방위산업", "국방과학기술", "사이버안보", "정보보안", "재난안전", "민방위",
            "국제안보협력", "군축", "평화유지", "위기관리", "안보이론", "전략연구"
        ],
        "question_types": ["정책설명", "제도소개", "전략분석", "기술개요", "협력체계", "이론적근거"]
    },
    "우주_위성": {
        "subcategories": [
            "우주탐사", "위성시스템", "발사체", "우주과학", "위성통신", "위성항법",
            "지구관측", "우주환경", "궤도역학", "우주추진", "우주정책", "우주법",
            "위성영상분석", "우주산업", "우주국제협력", "우주기술", "우주기상", "천체물리"
        ],
        "question_types": ["기술설명", "시스템구조", "운용방법", "정책내용", "과학원리", "응용분야"]
    },
    "윤리_정책": {
        "subcategories": [
            "연구윤리", "생명윤리", "의료윤리", "인공지능윤리", "정보윤리", "환경윤리",
            "과학기술정책", "혁신정책", "규제정책", "사회정책", "정책평가", "정책분석",
            "거버넌스", "공공윤리", "기업윤리", "전문가윤리", "데이터윤리", "디지털윤리"
        ],
        "question_types": ["윤리원칙", "정책목표", "규범내용", "적용사례", "쟁점분석", "개선방향"]
    }
}

# 질문 생성 프롬프트 템플릿
QUESTION_GENERATION_PROMPT = """당신은 전문 교육 콘텐츠 제작자입니다. 다음 조건에 맞는 고품질 QA 쌍을 생성해주세요.

**도메인**: {domain}
**세부 카테고리**: {subcategory}
**질문 유형**: {question_type}

**요구사항**:
1. 포멀하고 전문적인 어투로 작성
2. 질문은 명확하고 구체적으로 (50-150자)
3. 답변은 정확하고 상세하게 (200-600자)
4. 학술적/전문적 용어 적절히 사용
5. 실제 교육/업무에 활용 가능한 내용
6. 답변에는 논리적 근거와 설명 포함

**출력 형식** (JSON):
{{
  "question": "질문 내용",
  "answer": "답변 내용",
  "domain": "{domain}",
  "subcategory": "{subcategory}",
  "question_type": "{question_type}",
  "difficulty": "초급|중급|고급",
  "keywords": ["키워드1", "키워드2", "키워드3"]
}}

한 개의 QA 쌍만 생성하세요."""


class QADatasetGenerator:
    def __init__(self, output_dir: str = "qa_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_file = self.output_dir / "qa_data8.jsonl"
        self.log_file = self.output_dir / "generation_log8.txt"
        self.stats_file = self.output_dir / "statistics8.json"
        
        self.total_generated = 0
        self.stats = {domain: 0 for domain in DOMAIN_KEYWORDS.keys()}
        
    def log(self, message: str):
        """로그 기록"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
    
    async def generate_qa_pair(self, session: aiohttp.ClientSession, 
                               domain: str, subcategory: str, 
                               question_type: str) -> Dict[str, Any]:
        """단일 QA 쌍 생성"""
        prompt = QUESTION_GENERATION_PROMPT.format(
            domain=domain,
            subcategory=subcategory,
            question_type=question_type
        )
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        payload = {
            "model": "gpt-4o-mini",  # 비용 효율적
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 1000
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(API_URL, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        # JSON 추출
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0].strip()
                        
                        qa_data = json.loads(content)
                        qa_data["generated_at"] = datetime.now().isoformat()
                        return qa_data
                    
                    elif response.status == 429:  # Rate limit
                        wait_time = (attempt + 1) * 5
                        await asyncio.sleep(wait_time)
                    else:
                        self.log(f"API Error: {response.status}")
                        await asyncio.sleep(2)
                        
            except Exception as e:
                self.log(f"Error generating QA: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
        
        return None
    
    async def generate_batch(self, batch_size: int = 10):
        """배치 단위로 QA 생성"""
        tasks = []
        
        # 도메인 균등 분배
        domains = list(DOMAIN_KEYWORDS.keys())
        
        async with aiohttp.ClientSession() as session:
            for _ in range(batch_size):
                domain = random.choice(domains)
                domain_info = DOMAIN_KEYWORDS[domain]
                subcategory = random.choice(domain_info["subcategories"])
                question_type = random.choice(domain_info["question_types"])
                
                task = self.generate_qa_pair(session, domain, subcategory, question_type)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # 성공한 결과만 저장
            valid_results = [r for r in results if r is not None]
            
            with open(self.data_file, "a", encoding="utf-8") as f:
                for qa in valid_results:
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    self.stats[qa["domain"]] += 1
                    self.total_generated += 1
            
            return len(valid_results)
    
    async def generate_dataset(self, target_count: int = 500000, 
                               batch_size: int = 10,
                               save_interval: int = 1000):
        """전체 데이터셋 생성"""
        self.log(f"데이터셋 생성 시작: 목표 {target_count}건")
        self.log(f"배치 크기: {batch_size}, 저장 주기: {save_interval}")
        
        pbar = tqdm(total=target_count, desc="QA 생성 진행")
        
        try:
            while self.total_generated < target_count:
                generated = await self.generate_batch(batch_size)
                pbar.update(generated)
                
                # 통계 저장
                if self.total_generated % save_interval == 0:
                    self.save_statistics()
                    self.log(f"진행: {self.total_generated}/{target_count} ({self.total_generated/target_count*100:.1f}%)")
                
                # Rate limit 방지
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            self.log("사용자에 의해 중단됨")
        finally:
            pbar.close()
            self.save_statistics()
            self.log(f"최종 생성 개수: {self.total_generated}")
    
    def save_statistics(self):
        """통계 정보 저장"""
        stats = {
            "total_generated": self.total_generated,
            "domain_distribution": self.stats,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def load_existing_data(self):
        """기존 데이터 로드 (재시작 시)"""
        if self.data_file.exists():
            with open(self.data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        qa = json.loads(line)
                        self.total_generated += 1
                        self.stats[qa["domain"]] += 1
            self.log(f"기존 데이터 로드: {self.total_generated}건")


def main():
    """메인 실행 함수"""
    print("="*60)
    print("한국어 QA 데이터셋 생성 시스템")
    print("="*60)
    
    # API 키 확인
    if OPENAI_API_KEY == "your-api-key-here":
        print("\n⚠️  OpenAI API 키를 설정해주세요:")
        print("export OPENAI_API_KEY='your-actual-api-key'")
        return
    
    # 설정
    target_count = int(input("\n생성할 QA 개수 (기본: 500000): ") or "500000")
    batch_size = int(input("배치 크기 (기본: 10, 권장: 5-20): ") or "10")
    
    # 생성기 초기화
    generator = QADatasetGenerator()
    generator.load_existing_data()
    
    print(f"\n📊 현재 진행 상황: {generator.total_generated}건")
    print(f"🎯 목표: {target_count}건")
    print(f"📦 배치 크기: {batch_size}")
    print(f"💾 저장 위치: {generator.output_dir}")
    
    input("\n시작하려면 Enter를 누르세요...")
    
    # 비동기 실행
    asyncio.run(generator.generate_dataset(target_count, batch_size))
    
    print("\n✅ 데이터셋 생성 완료!")
    print(f"📁 파일 위치: {generator.data_file}")
    print(f"📊 통계 파일: {generator.stats_file}")


if __name__ == "__main__":
    main()
