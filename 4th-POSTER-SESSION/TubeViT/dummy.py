import torch

# .pt 파일 로드
model_state = torch.load('tubevit_b_01.pt')

# 새로운 키와 값을 상태 사전에 추가
new_key = 'epoch'
new_value = '10'  # 새로운 값 설정
model_state[new_key] = new_value

# 변경된 상태 사전을 .pt 파일로 저장
torch.save(model_state, 'tubevit_b_01.pt')
