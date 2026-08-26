# IsalSR — Code Ocean compute capsule environment.
#
# Only base images from Code Ocean's own registry are accepted, and which tags a
# deployment offers is a property of that deployment.  So this file assumes as
# little about the base as possible: postInstall installs the system packages,
# Python 3.11 and the C++ toolchain itself, at a private prefix that cannot
# collide with a base image that already carries conda.  Changing the FROM line
# from the Environment Editor is therefore safe -- nothing else needs to follow.
#
# The tag below is Ubuntu 20.04 with Python 3.9, which is neither what the
# project needs (>= 3.11) nor able to build the C++ engine on its own (GCC 9,
# and CMakeLists.txt targets -march=x86-64-v3, which GCC did not accept before
# 11).  It is used precisely because it is the oldest base in play: postInstall
# supplies the interpreter and the compiler, so if the capsule works here it
# works on any newer starter environment too.
#
# Everything must be installed at build time: a Reproducible Run has no network.
# The single exception is the capsule's own package, which cannot be installed
# now because postInstall runs before Code Ocean mounts /code; code/run compiles
# it from source during the run.

FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# The capsule's interpreter takes precedence over the base image's.  code/run
# does not depend on this -- it resolves the interpreter by absolute path -- so
# regenerating this file from the Environment Editor cannot break a run.
ENV PATH="/opt/isalsr-conda/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /
COPY postInstall /
RUN /postInstall
