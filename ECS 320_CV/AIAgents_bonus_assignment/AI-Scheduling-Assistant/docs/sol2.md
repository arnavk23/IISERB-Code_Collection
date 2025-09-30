1. based on email content, extract the time window and duration of the proposed meeting using an LLM and convert this into a timestamp format
2. use the calendar retriever function to get the calendars of all the attendees for that time window
3. algorithmically search if a slot of duration is available in the time. You might need a calendar parser and searcher utility class for this.
4. if slot found, good. 
5. if slot not found, ask LLM to rank the attendee's meetings in the window and proposed meeting
6. meetings with summary(lowercase): 'weekend' and 'off hours' cannot be rescheduled and are off-limits, programmatically ignore them, don't give them to LLMs context.
7. based on these priorities again search for a slot. starting from lowest priority you can now override meetings. prefer free times


HIP_VISIBLE_DEVICES=0 vllm serve deepseek-ai/deepseek-llm-7b-chat         --gpu-memory-utilization 0.9         --swap-space 16         --disable-log-requests         --dtype float16         --max-model-len 10000         --tensor-parallel-size 1         --host 0.0.0.0         --port 3000         --num-scheduler-steps 10         --max-num-seqs 128         --max-num-batched-tokens 10000         --max-model-len 10000         --distributed-executor-backend "mp"

HIP_VISIBLE_DEVICES=0 vllm serve Models/meta-llama/Llama-3.3-70B-Instruct \
    --disable-log-requests \
    --dtype bfloat16 \
    --max-model-len 10000 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 4000 \
    --num-scheduler-steps 10 \
    --distributed-executor-backend "mp" \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --use-v2-block-manager \
    --kv-cache-dtype fp8 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.95 \
    --swap-space 0