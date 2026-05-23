# Differential Privacy against a Membership Inference Attack

This repository contains the code and supporting files for a project evaluating the effectiveness of Differential Privacy (DP) as a defense against **Membership Inference Attacks (MIA)**. A detailed analysis of the work, including methodology and experimental results, can be found in the **`article.pdf`** file.

## Repository Structure

- **`target.py`**: Script to create and train the *vulnerable* target model (without defense mechanisms).
- **`target_DP.py`**: Script to train the target model *with Differential Privacy implemented*, useful for mitigating exposure risks.
- **`attack.py`**: Script that executes the Membership Inference Attack against the standard target model (`target.py`) to demonstrate its vulnerability.
- **`attack_DP.py`**: Script that executes the Membership Inference Attack on the protected model (`target_DP.py`) to test and quantify the protective effectiveness of Differential Privacy.
- **`requirements.txt`**: File containing all the software dependencies and libraries (e.g., PyTorch, Opacus, etc.) needed to run the project.
- **`article.pdf`**: The main document that theoretically describes the approach and presents the results of this work (refer to it for in-depth details).
- **`technical_documentation.pdf`**: In-depth technical details about the implementation, model architectures, and the mathematical framework behind the Differential Privacy defense.
- **`presentation.pdf`**: Slides summarizing the project's context, the proposed methodology, and key findings for a quick overview.

## Requirements and Execution

It is recommended to create a virtual environment (e.g., conda or venv) and install the dependencies as follows:

```bash
pip install -r requirements.txt
```