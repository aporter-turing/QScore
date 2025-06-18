FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

FROM base AS final
# Install precompiled ANTs binaries 
COPY --from=829077602501.dkr.ecr.us-east-1.amazonaws.com/research/dependency/ants:2.5.0 /opt/ants/bin /opt/ants/bin
ENV PATH="/opt/ants/bin:$PATH"

RUN apt-get update && \
    apt-get install -y wget file nano dcm2niix python3 python3-pip strace

RUN apt-get install -y libgl1-mesa-dev && \
    wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
    python3 fslinstaller.py -d /opt/fsl

RUN pip install nipype nibabel scipy numpy watchdog pydicom
# FSL env variables
ENV FSLDIR=/opt/fsl
ENV FSL_DIR=/opt/fsl
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV FSLMULTIFILEQUIT=TRUE
ENV FSLTCLSH=${FSLDIR}/bin/tclsh
ENV FSLWISH=${FSLDIR}/bin/wish
ENV FSLGECUDAQ=cuda.q
ENV FSL_LOAD_NIFTI_EXTENSIONS=0
ENV FSL_SKIP_GLOBAL=0
ENV OPENBLAS_NUM_THREADS=1
ENV PATH=${FSLDIR}/share/fsl/bin:${PATH}
ENV USER=""

WORKDIR /app
COPY . /app/

ENV QSCORE_PATH /app/q_score
ENV DATA_DIR /incoming
ENV NIFTI_DIR /incoming

CMD ["python3","-u", "main.py"]