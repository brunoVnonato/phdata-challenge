# Official miniconda image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the entire project to the working directory
COPY . .

# Create conda environment into the container
RUN conda env create -f conda_environment.yml

# Make the environment active by default
RUN echo "source activate housing" > ~/.bashrc
ENV PATH=/opt/conda/envs/housing/bin:$PATH

# Expose the port that FastAPI will run on
EXPOSE 8080

# Command to run the FastAPI application using uvicorn server
CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port", "8080"]
