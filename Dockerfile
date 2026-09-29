FROM nvcr.io/nvidia/tritonserver:25.11-py3

RUN python3 -m pip install --no-cache-dir \
    scikit-learn==1.8.0 \
    joblib==1.5.3 \
    numpy==2.3.5 \
    scipy==1.16.3 \
    threadpoolctl==3.6.0

CMD ["tritonserver", "--model-repository=/models"]
