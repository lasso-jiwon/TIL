* 아나콘다 버전 확인하기

  conda -V

* 가상황경에 파이썬 설치하기

  conda create -n ToDoList python=3.8.8 anaconda

* 생성된 가상환경 리스트 확인

  conda info --envs

* 생성된 가상환경 활성화

  conda activate ToDoList

* 생성된 가상환경 비활성화

  conda deactivate

* 가상환경 삭제

  conda remove -n ToDoList --all

* (django)장고를 ToDoList 가상환경에 설치

  conda install django

* django project 생성

  django-admin startproject myproject

* server 실행

  python manage.py runserver

___

# django에서 패턴 (page 142)

Model           Controller      View = MVC

models.py    views.py        templares = MTV



application 생성

$ python manage.py startapp my_to_do_app





포트폴리오 팀별로 작성하여 조장이 대표로 제출