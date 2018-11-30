from setuptools import setup, find_packages
setup(
    name="The Awesome Digit Recognizer",
    version="0.1",
    packages=find_packages(),
    scripts=['digit_recognizer/digitrec.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['docutils>=0.3'],

    # metadata to display on PyPI
    author="Daire Ni Chathain",
    author_email="dairenichat@gmail.com",
    description="This package recognizes handrawn digits",
    url="https://github.com/DaireNiC/Emerging-Techologie",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/DaireNiC/Emerging-Techologies/issues",
        "Source Code": "https://github.com/DaireNiC/Emerging-Techologies/",
    },
    entry_points = {
        'console_scripts': ['digitrec = digit_recognizer.digitrec.__main__:main'],
    }


)
