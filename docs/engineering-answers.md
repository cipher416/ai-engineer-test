# Engineering question answers

## 1. REST API versus MCP in AI systems

REST APIs usually organize access around resources, using specific HTTP methods for different operations. For example, GET /orders/{id} retrieves an order and POST /orders creates one.

MCP gives AI applications and clients a standardized way to discover and use tools, resources and prompts. This reduces the integration work needed to connect a certain system with an agent.

A system can use both. An MCP server can expose selected operations from its existing REST API, while the REST API also supports general application integration.

## 2. How REST and MCP improve AI use cases

An AI agent can use custom tools that call REST endpoints, or connect to an MCP server to access local or remote tools. These tools can access an application/platform's data or modify data in a system. Some example use cases would be searching internal documentation or submitting a support request.

## 3. Ensuring an AI agent answers correctly

We can check an agent's answers against the expected results for its task. For factual questions, this means checking whether the answer is supported by evidence and whether the citations are accurate. For actions, we also need to check the tool selected, its arguments and the result of the operation. A valid JSON response can still contain an incorrect answer.

The agent can use trusted sources for facts and call code to handle calculations and business rules. Tool inputs and outputs should be validated, and users should be able to see the evidence behind an answer. If information is missing, the agent should ask for clarification or explain what it cannot determine. Tool and provider errors need to be reported. Decisions that mutate data/have an impact or a blast radius should also require a human in the loop, as AI is inherently non-deterministic.

Testing should use representative examples with reviewed expected results. These should include ambiguous requests, missing information, tool failures and malicious instructions in retrieved content. We should keep a separate test set for evaluation, with no duplicate or closely related examples from the development set. Checking retrieval, tool use and the final answer separately helps identify where an error occurred.

Some evaluation framework options are:

- [Ragas](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/) can evaluate RAG systems using metrics such as context precision, context recall and faithfulness. Faithfulness checks whether the answer is supported by the retrieved context.

- [DeepEval](https://deepeval.com/docs/getting-started) integrates with pytest, so evaluations can run alongside other Python tests. It also has metrics for agent task completion.

- [Promptfoo](https://www.promptfoo.dev/docs/configuration/expected-outputs/) supports exact-match checks, JSON validation and custom assertions. It can also use a model to grade responses against a rubric.

For known results, such as a total or tool argument, we can use exact checks. An LLM judge can evaluate qualities that are harder to check in code, but we should compare its ratings with human ratings. When comparing changes, use the same evaluation data and grading criteria, and account for the cost of the judge calls.

We should rerun evaluations after changing the model, prompts or tools. In production, we also need to monitor new failures, accuracy, latency and cost. Passing these evals does not necessarily mean that every future answer is guaranteed to be correct.

## 4. Docker and containerization in AI

Docker packages an application with its runtime and dependencies. The same image can be used in development, CI and deployment, which reduces differences between environments. Pinning dependencies and versioning images helps reproduce builds and allows a previous image to be used for rollback.

An AI application may have a web API, model server, background workers and a database. These can run in separate containers and scale based on their compute needs. Persistent data can be stored in volumes or external storage, with credentials supplied when the container starts. Large model weights can be managed separately as versioned artifacts, so they do not need to be copied into every application build.

For example, we can use vLLM's official vllm/vllm-openai image to serve a supported model through an OpenAI-compatible API. The application calls the model server over the container network. The vLLM container gets GPU access and a mounted cache for downloaded model weights. Pinning the image and model revision keeps the versions consistent across redeployments. The [vLLM Docker guide](https://docs.vllm.ai/en/latest/deployment/docker/) shows this setup.

GPU inference still requires compatible drivers and runtime configuration on the host, as well as enough GPU memory for the model and workload. vLLM also uses shared memory between processes, particularly when using tensor parallelism across GPUs. The container needs enough shared memory for this, as covered in the [vLLM shared-memory guidance](https://docs.vllm.ai/en/latest/deployment/docker/#pre-built-images). We should set memory and compute limits for the workload and use health checks and logs to identify failed or overloaded services.

## 5. Fine-tuning an LLM from raw data

Fine-tuning adapts a pretrained model to a task or response style. Training a model from random weights is pretraining, while further language-model training on a domain corpus is continued pretraining. For supervised fine-tuning, we need examples of the input and the response the model should produce. Raw documents need to be converted into these examples before training.

Before training, we should define the behavior we want and measure how the existing model performs. We can try prompting or retrieval to see whether they solve the problem. Fine-tuning can help the model learn consistent behavior, while retrieval is usually better for business facts that change frequently.

The data needs to be cleaned and deduplicated, with unnecessary personal information removed. We also need permission to use it. Split it into training, validation and held-out test sets, keeping related source documents together to avoid leakage. Review the expected outputs and include difficult cases, such as how to respond when information is missing. Image tasks need image examples and a compatible vision-language model.

The model should fit the task, license and deployment budget. Format the examples using its tokenizer and expected input format, and account for sequence limits. We can update the full model or use parameter-efficient adapters such as LoRA. Validation results can guide the learning rate, batch size and training duration.

After training, compare the selected model with the baseline on held-out data. Check task quality, regressions and privacy risks, along with latency and resource use. Training loss alone does not show whether the model is useful. Keep versions of the model, dataset and configuration, then deploy gradually and monitor the results. The previous version should remain available for rollback.
