# git command

> 기본 명령어를 정리합니다.





##### init

* `git init` 이라고 작성합니다

* `.git` 폴더를 생성해주는 명령어. 처음 한번만 실행합니다.



##### touch

* `touch 파일명.확장자` : 지금 폴더에 파일을 생성해줌
  * 예) `touch a.txt` : a 이름의 텍스트 파일 생성



##### add

Working directory 에 있는 파일을 staging area에 올리는 명령어.

* 기본 사용법
  * file name에 `.`을 입력하면 모든 파일, 폴더를 한번에 올림
  * 예1) `git add a.txt` a라는 파일을 올림
  * 예2) `git add .` 모든 파일, 폴더를 한번에 올림

```bash
git add <file name>
```



##### git status

* `git status` git의 현재 상태를 확인하는 명령어
* 파일을 add해서 올리면 올린 파일을 보여줌
* 기본 명령어

```bash
git status
```



##### commit

staging area에 있는 파일들을 하나의 commit으로 저장하는 명령어

* 기본 사용법
  * `-m` : 커밋 메시지를 작성하기 위한 옵션

``` bash
git commit -m "message"
```



##### remote

원격저장소를 관리하기 위한 명령어

* add : 원격 저장소를 추가하는 명령어

```bash
git remote add origin <remote name> <URL>
```

* remove : 원격 저장소를 제거하는 명령어

``` bash
git remote remove <remote name>
```



##### push

로컬에 저장되어 있는 커밋들을 원격저장소에 업로드 하는 명령어

* 기본 사용 방법

``` bash
git push origin master
```





___



# 요약

**add - commit - push** 순서를 기억하자!

git add .

git commit -m "update"

git push origin master 
