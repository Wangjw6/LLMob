<a href='https://arxiv.org/abs/2304.04370'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/agiresearch/OpenAGI/blob/main/LICENSE)


# Large Language Models as Urban Residents: An LLM Agent Framework for Personal Mobility Generation

## Description
This repository is the implemetation of LLMob from [Large Language Models as Urban Residents: An LLM Agent Framework for Personal Mobility Generation](https://arxiv.org/abs/2402.14744). 
LLMob is a simple framework that takes advantage of Large Language Models (LLMs) for personal activity trajectory generation.

<p align="center">
<img src="images/LLMob.png">
</p>

<p align="center">
<img src="images/demo.png">
</p>


## Key Components
- **./simulator/engine/person.py**: Generate personal activity trajectory according to real-world check-in data.
- **./simulator/engine/functions/traj_infer.py**: Personal activity trajectory generation function.
- **./simulator/engine/functions/PISC.py**: Personal activity pattern identification function.
- - **./simulator/engine/memory/retrieval_helper.py**: Function related to motivation retrieval.
- **./simulator/prompt_template**: Prompt template used in this project.

## Usage

To get started with LLM Explorer, follow these steps:

```bash
git clone https://github.com/yourusername/llm-explorer.git
cd llm-explorer
conda env create -f environment.yml
conda activate llm
```

## Citation

If you find our work useful in your research, please cite our [paper](https://arxiv.org/abs/2402.14744):

```
@article{wang2024large,
  title={Large Language Models as Urban Residents: An LLM Agent Framework for Personal Mobility Generation},
  author={Wang, Jiawei and Jiang, Renhe and Yang, Chuang and Wu, Zengqing and Onizuka, Makoto and Shibasaki, Ryosuke and Xiao, Chuan},
  journal={arXiv preprint arXiv:2402.14744},
  year={2024}
}
```

## Acknowledgments

This project refers to several open-source ChatGPT application:

- [Generative Agents: Interactive Simulacra of Human Behavior](https://github.com/joonspk-research/generative_agents)

- [MetaGPT](https://github.com/geekan/MetaGPT/tree/main)

The raw data used in this project is from [Foursquare API](https://location.foursquare.com/developer/).
