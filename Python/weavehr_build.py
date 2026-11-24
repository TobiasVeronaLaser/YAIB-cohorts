"""Build the WeavEHR project read by src/weavehr.py: WeavEHR's extraction and concept steps for one
dataset, written to `weavehr_project_path` in config.yaml. Requires WeavEHR (Python >= 3.13).

All dataset and concept configs are WeavEHR's own, including the ricu-compatible datasets (*-ricu)
for comparisons with the cohorts extracted with ricu.
"""
import argparse
import tempfile
from pathlib import Path

import yaml
from weavehr import ConceptStep, ExtractionStep, WeavEHRProject
from weavehr.logging import configure_logging
from weavehr import load_configs

from src import conf

with open('../weavehr-extensions/config.yml') as stream:  # this path is relative to the calling environment
    SOURCES = list(yaml.safe_load(stream)['sources'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, choices=SOURCES,
                        help='name of the WeavEHR dataset, e.g., mimic-iv or mimic-iv-demo')
    parser.add_argument('--version', required=True, help='version of the dataset, e.g., 2.2')
    parser.add_argument('--data-path', required=True, help='directory with the raw data files of the dataset')
    parser.add_argument('--overwrite', action='store_true', help='rebuild steps that have already been run')
    args = parser.parse_args()

    dataset = {'name': args.src, 'version': args.version}
    steps = {
        'extraction': {'name': 'Extraction', 'version': '1.0.0', 'overwrite': args.overwrite,
                       'config': {'data': [{**dataset, 'path': str(Path(args.data_path).resolve())}]}},
        'concept': {'name': 'Concept', 'version': '1.0.0', 'overwrite': args.overwrite,
                    'config': {'extraction_step': 'Extraction', 'mapping_configs': [dataset]}},
    }

    configure_logging(level="INFO")
    load_configs(dataset_filter=[args.src])
    project_path = Path(conf['weavehr_project_path'].format(src=args.src))
    with tempfile.TemporaryDirectory() as config_dir, WeavEHRProject(project_path) as project:
        for name, step in steps.items():
            with open(Path(config_dir) / f'{name}.yml', 'w') as stream:
                yaml.safe_dump(step, stream)
        ExtractionStep.load(project, Path(config_dir) / 'extraction.yml').run()
        ConceptStep.load(project, Path(config_dir) / 'concept.yml').run()
    print(f'WeavEHR project for {args.src} written to {project_path}')
