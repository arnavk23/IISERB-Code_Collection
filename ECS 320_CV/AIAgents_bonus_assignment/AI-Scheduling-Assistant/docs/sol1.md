You are given an input meeting scheduling request

1. pull the calendars for the next 1 month for the attendee from the request date
2. format all of it into 1 prompt
3. format the meeting agenda, subject and other data from the request into the prompt
3. pass this prompt to an LLM call and ask it to propose a slot for the meeting
4. it should give priority to free slots
5. if no free slots are available, use smeantic understanding of the attendee's meetings to decide which can be rescheduled(eg unimportant ones)
6. fix the meeting

HIP_VISIBLE_DEVICES=0 vllm serve /home/user/Models/deepseek-ai/deepseek-llm-7b-chat \
        --gpu-memory-utilization 0.9 \
        --swap-space 16 \
        --disable-log-requests \
        --dtype bfloat16 \
        --max-model-len 4069 \
        --tensor-parallel-size 1 \
        --host 0.0.0.0 \
        --port 3000 \
        --num-scheduler-steps 10 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 4069 \
        --max-model-len 4069 \
        --distributed-executor-backend "mp"

HIP_VISIBLE_DEVICES=0 vllm serve /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B \
        --gpu-memory-utilization 0.9 \
        --swap-space 16 \
        --disable-log-requests \
        --dtype float16 \
        --max-model-len 10000 \
        --tensor-parallel-size 1 \
        --host 0.0.0.0 \
        --port 3001 \
        --num-scheduler-steps 10 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 10000 \
        --max-model-len 10000 \
        --distributed-executor-backend "mp"