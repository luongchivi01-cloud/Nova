# Intellectual Property Declaration

This document records the project's intellectual-property position for the
HackAIthon submission.

## Project code

The competition-specific orchestration, JSON/CSV handling, risk routing,
checkpointing, strict validation, prompts, and Docker configuration in this
repository are submitted as the team's implementation.

No third-party repository source tree is committed to this GitHub repository.
The final Docker image contains the project source, the dependencies installed
by its Dockerfile, and the attributed Qwen GGUF model.

## Third-party components

Third-party components and their licenses are listed in
`THIRD_PARTY_NOTICES.md`. License copies are stored under `licenses/`.

References to optional projects in compatibility modules do not mean those
projects are installed or included in the final image. The official CPU profile
disables RAG, vendor RAG fusion, VnCoreNLP, Transformers, vLLM, CUDA, and GPU
execution.

## Data and generated results

The organizer-provided public-test dataset and generated leaderboard
predictions are deliberately excluded from GitHub. They are processed only for
the competition submission workflow.

## Accuracy of submission information

The model identity, checksum, Docker tag, Docker digest, runtime constraints,
and validation claims are recorded in the repository so they can be checked by
the organizer.
