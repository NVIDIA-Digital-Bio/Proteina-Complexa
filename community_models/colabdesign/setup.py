from setuptools import setup, find_packages

setup(
    name='colabdesign',
    version='1.1.1.1',
    description='Local modified version of ColabDesign',
    long_description="Making Protein Design accessible to all via Google Colab! (Local version)",
    long_description_content_type='text/markdown',
    packages=find_packages(include=['colabdesign*']),
    # Dependencies are installed separately via pip colabdesign==1.1.1
    # then this local version replaces it
    install_requires=[],
    include_package_data=True
)
