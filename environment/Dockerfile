# IsalSR — Code Ocean compute capsule environment.
#
# Everything the capsule needs is baked in here, at build time, because a
# Reproducible Run has NO network access.  Nothing may be downloaded later.
#
# The one thing that cannot be baked in is the capsule's own package: the
# post-install script runs during the build phase and Code Ocean does not mount
# /code or /data at that point.  `isalsr` is therefore compiled from /code by
# code/run, offline, against the build backend installed below.
#
# Base image: registry.codeocean.com/codeocean/ubuntu:22.04 provides GCC 11.4
# (the C++ engine targets -march=x86-64-v3, which GCC did not accept before 11)
# and CMake 3.22 (pyproject requires >= 3.18).  Python comes from Miniforge in
# postInstall, pinned to 3.11 to match the environment the reported results were
# produced on; Ubuntu 22.04's own Python is 3.10 and the project requires >=3.11.
#
# If you change the base image from the Environment Editor, Code Ocean rewrites
# the FROM line and leaves the rest of this file intact.  Any Ubuntu 22.04 or
# 24.04 base works; a base whose GCC is older than 11 does not.

FROM registry.codeocean.com/codeocean/ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        graphviz \
        libgraphviz-dev \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /
COPY postInstall /
RUN /postInstall
