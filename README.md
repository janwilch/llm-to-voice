# llm-to-voice

Contains a pure C/C++ implementation of a Prompt-LLM-TTS pipeline (a Python prototype will be deleted later). A client application can input a string prompt and receives a streamed string response plus a streamed vocalization (mono audio, 24kHz).

## Data Flows

The following shows the rough overall procedure in pseudo code.

```
// before: create & warmup

llmvoice_set_style:
  // replace context & memory w/ fresh chat template
  ILlmBackend.createFreshContext(systemPrompt)
  // define speaker for following generations
  ITtsBackend.createFreshContext(seed, instruct)

llmvoice_submit:
  INIT llmTokenQueue, segmentsToTtsQueue, segmentsOutQueue, pcmOutQueue

  THREAD1 (while not llmTokenQueue.closed):
    ILlmBackend.synthesizeToQueue(userMessage, llmTokenQueue)

  THREAD2 (while not llmTokenQueue.empty):
    token <- llmTokenQueue.pop()
    segment <- accumulate(token).segment()   // batches cleaned segments (e.g. w/o <think>)
    segmentsToTtsQueue.push(segment)
    segmentsOutQueue.push(segment)

  THREAD3 (while not segmentsToTtsQueue.empty && closed):
    segment <- segmentsToTtsQueue.pop()
    ITtsBackend.synthesizeToQueue(segment, pcmOutQueue)

llmvoice_cancel:
  llmTokenQueue.close()
  segmentsToTtsQueue.close()
  segmentsOutQueue.close()
  pcmOutQueue.close()

llmvoice_poll_text:
  RETURN segmentsOutQueue.get(requestedBytes)

llmvoice_poll_pcm:
  RETURN pcmOutQueue.get(requestedBytes)

llmvoice_is_done:
  RETURN segmentsOutQueue.closed && empty AND pcmOutQueue.closed && empty

// after: destroy
```

## C ABI

This is exposed via a C ABI like so:

```c
typedef struct llmvoice_handle llmvoice_handle;

typedef struct {
  const char* llm_model_path;
  const char* tts_talker_path;
  const char* tts_codec_path;
  int         sample_rate_hint;   // 0 = native 24000
  int         llm_context_size;   // 0 = default 8192
} llmvoice_config;

llmvoice_handle* llmvoice_create(const llmvoice_config* cfg);
void llmvoice_warmup(llmvoice_handle*);
void llmvoice_destroy(llmvoice_handle*);

void llmvoice_set_style(
  llmvoice_handle*,
  const char* system_prompt_utf8,
  const int64_t speaker_seed,
  const char* speaker_instruct);

void llmvoice_submit(llmvoice_handle*, const char* prompt_utf8);
void llmvoice_cancel(llmvoice_handle*);

int llmvoice_poll_text(llmvoice_handle*, char* dst_utf8, int max_bytes);
int llmvoice_poll_pcm(llmvoice_handle*, float* dst, int max_frames);
int llmvoice_is_done(llmvoice_handle*);
```
