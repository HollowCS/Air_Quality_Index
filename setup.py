from setuptools import find_packages, setup


HYPEN_E_DOT = "-e ."


def get_requirements(file:str) -> list[str] :
    required_file = []
    with open(file) as file_obj:
        required_file = file_obj.readlines()
        [req.replace("\n", "") for req in required_file]

        if HYPEN_E_DOT in required_file:
            required_file.remove(HYPEN_E_DOT)
    return required_file


setup(
    name="Air Quality Index",
    version="0.0.1",
    author="Madhu Sudan Mandal",
    author_email="madhusudan7848@gmail.com",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages()
)
