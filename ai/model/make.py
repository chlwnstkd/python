from transformers import GPT2Config, GPT2Model

# config.json 파일 로드
config = GPT2Config.from_json_file('sensibility/config.json')

# 파라미터 파일 로드
model = GPT2Model.from_pretrained('sensibility/pytorch_model.bin', config=config)

# 모델 저장
config.save_pretrained('new_model_directory')
model.save_pretrained('new_model_directory')