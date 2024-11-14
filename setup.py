from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """This function returns a list of requirements from the file."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        # Remove '-e .' if present in requirements
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='My_AI_Project_structure',
    version='0.0.1',
    author='Mohamed',
    author_email='mohamedadel.msk@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages(),
)
