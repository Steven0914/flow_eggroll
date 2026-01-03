import wandb
import pandas as pd

# 1. API 객체 초기화
api = wandb.Api()

# 2. 특정 Run 객체 가져오기 (경로 정확히 입력)
# runs/isdk51wu 처럼 ID까지 포함된 경로는 api.run()을 씁니다.
run = api.run("hyunuk/flow_eggroll/runs/isdk51wu")

print(f"다운로드 중: {run.name} ({run.id})")

# 3. 히스토리(Charts/Train 데이터) 가져오기
# samples=1000000 : 기본값은 500개만 가져오므로, 전체 데이터를 위해 숫자를 크게 잡습니다.
history_df = run.history(samples=1000000) 

# (선택사항) Config 정보도 같은 CSV에 넣고 싶다면 아래 주석 해제
# for k, v in run.config.items():
#     if not k.startswith('_'):
#         history_df[f"config_{k}"] = v

# 4. CSV 저장
filename = f"wandb_data_{run.id}.csv"
history_df.to_csv(filename, index=False)

print(f"추출 완료! '{filename}' 파일을 확인하세요.")
print(f"총 {len(history_df)}개의 Step 데이터가 저장되었습니다.")