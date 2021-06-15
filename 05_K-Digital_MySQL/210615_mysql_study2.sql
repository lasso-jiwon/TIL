# database 변경
use classicmodels;

SELECT 1+2;

SELECT 1+2 FROM dual;

# 데이터 검색: SELECT문
/*
SELECT 절
FROM 절
WHERE 절
group by 절
Having 절
ORDER by 절

[해석순서]
FROM 절
WHERE 절
group by 절
Having 절
SELECT 절
ORDER by 절
*/

# 고객 테이블에서 해당하는 컬럼의 정보만 검색
SELECT customerNumer, customerName, phone, city
FROM customers;

# 고객 전체의 정보를 검색하라.
SELECT * FROM customers;

SELECT city FROM customers;

# distinct 중복제거
SELECT distinct city FROM customers;

SELECT count(city) FROM customers;

SELECT count(customerNumber) FROM customers;

SELECT count(customerNumber),count(city) FROM customers;

SELECT amount, amount*2 FROM payments;
SELECT amount, amount*2 AS bonus FROM payments;

SELECT *
FROM orderdetails
WHERE priceeach > 30;

SELECT *
FROM orderdetails
WHERE priceeach = 30;

SELECT *
FROM orderdetails
WHERE priceeach >= 30 AND priceeach <=50;

SELECT *
FROM orderdetails
WHERE priceeach BETWEEN 30 AND 50;


SELECT customerNumber
FROM customers
WHERE country='USA' OR country='Canada';

SELECT customerNumber
FROM customers
WHERE country IN ('USA', 'Canada');


SELECT customerNumber, country
FROM customers
WHERE country NOT IN ('USA', 'Canada');


SELECT customerNumber, country
FROM customers
WHERE NOT(country='USA' OR country='Canada');


# reportsTO컬럼의 값이 NULL일 때
# 값이 Null일 때는 IS로 비교해야한다. ( reportsTO = Null은 안됨.)
SELECT employeeNumber, firstName, reportsTO
FROM employees
WHERE reportsTO IS NULL;


# reportsTO컬럼의 값이 NULL이 아닐때
SELECT employeeNumber, firstName, reportsTO
FROM employees
WHERE reportsTO IS NOT NULL;

SELECT addressLine1
FROM customers
WHERE addressLine1 LIKE '%ST%';  # %는 모든을 의미(와일드카드 기호, ST가 포함이 된 모든)

SELECT addressLine1
FROM customers
WHERE addressLine1 LIKE '_T%'; # _는 하나를 의미(T앞에 어떤문자 하나가 있는 단어가 포함된 모든)


# 대소문자 구분해서 검색
SELECT addressLine1
FROM customers
WHERE addressLine1 LIKE BINARY '%ST%'; 

SELECT addressLine1
FROM customers
WHERE addressLine1 LIKE BINARY '_T%'; # _는 하나를 의미

SELECT count(customerNumber), city
FROM customers;


SELECT count(customerNumber), city
FROM customers
group by city;


SELECT sum(CASE WHEN city='Nantes' THEN 1 ELSE 0 END)  cnt 
FROM customers;


SELECT sum(CASE WHEN city='Nantes' THEN 1 
                WHEN city='Las Vegas' THEN 2
                WHEN city='Stavern' THEN 3
                ELSE 0 END)  cnt 
FROM customers;


# 다중 조건문
SELECT CASE WHEN city='Nantes' THEN 1 
                WHEN city='Las Vegas' THEN 2
                WHEN city='Stavern' THEN 3
                ELSE 0 END  cnt   # AS cnt에서 AS 생략
FROM customers;

# count(*) : 무조건 전체 레코드 수를 리턴한다.
# count(컬럼명) : NULL이 아닌 전체 레코드수를 리턴한다.
SELECT count(employeeNumber), count(reportsTO), count(*)
FROM employees;


SELECT SUM(CASE WHEN country='USA' THEN 1 ELSE 0 END) AS N_USA,
	   SUM(CASE WHEN country='USA' THEN 1 ELSE 0 END)/COUNT(*) AS USA_PORTION
FROM customers;


# JOIN
SELECT * FROM orders LEFT JOIN customers        # orders는 모두, customers 테이블은 공통적인 것만 가져옴.
ON orders.customerNumber = customers.customerNumber;


SELECT o.orderNumber, o.customerNumber, c.customerNumber 
FROM orders o LEFT JOIN customers c
ON o.customerNumber = c.customerNumber;

SELECT o.orderNumber, o.customerNumber, c.customerNumber 
FROM orders o INNER JOIN customers c
ON o.customerNumber = c.customerNumber;


SELECT buyprice
FROM products
ORDER by buyPrice ASC;


SELECT buyprice
FROM products
ORDER by buyPrice DESC;


SELECT buyprice,
	   row_number() over(ORDER BY buyprice) AS rownumber,
	   rank() over(ORDER BY buyprice) AS rnk,
	   dense_rank() over(ORDER BY buyprice) AS densrank
FROM products;

# subquery
/*
1. subquery는 ( )내에서 작성한다.
2. in: subquery의 결과가 여러개 일 때
   =, >=, <= : subquery의 결과가 1개 이하일때
*/

SELECT customernumber
FROM customers; # 125, 169, 206, 223, ....

SELECT ordernumber, customerNumber
FROM orders;

SELECT ordernumber
FROM orders
WHERE customerNumber= (
						SELECT customernumber
						FROM customers
                        WHERE customernumber=103);
                        

SELECT customernumber
FROM customers
WHERE city='NYC';

SELECT ordernumber
FROM orders
WHERE customerNumber in (
						SELECT customernumber
                        FROM customers
                        WHERE city='NYC');
                        


# table 생성
create table dept(
num int(3),
dname varchar(50),
dloc varchar(30)
);


# 데이터 추가
INSERT INTO dept(num, dname, dloc)
VALUES(10, 'sales', 'seoul');

INSERT INTO dept
VALUES(20, 'human', 'inchon');

INSERT INTO dept(num, dname, dloc)
VALUES(30, Null, NULL);

INSERT INTO dept(num, dloc)
VALUES(40, 'busan');

# 수정
UPDATE dept
SET dname='management', dloc='gangneung'
WHERE num=30;

# 삭제
DELETE FROM dept
WHERE num=40;

SELECT * FROM dept;


# create database
SHOW DATABASES;

CREATE DATABASE shop;

use shop;

CREATE TABLE person(
id int(10) primary key,
name varchar(3),
age int(3));


INSERT INTO person(id, name, age)
VALUES(20, '유대위', 35);

INSERT INTO person(id, name, age)
VALUES(30, '고수', 40);

SELECT * FROM person;

/*
SELECT schema_name , default_character_set_name
FROM information_schema.schemata;

ALTER DATABASE shop DEFAULT CHARACTER SET utf8;
ALTER SCHEMA 'shop' DEFAULT COLLATE utf8mb4_general_ci;

ALTER TABLE 'shop'.'person'
CHARACTER SET = utf8;
*/
UPDATE person
SET age=30
WHERE id=20;



### 날짜 함수
SELECT now(), curdate(), current_date, current_date();

# yyyy-mm-dd 형태를 yyyymmdd인 숫자형태로 출력
SELECT curdate(), curdate()+0, curdate()+1;

# 날짜 사이의 일 수 알기
SELECT datediff(curdate(), '2021-06-02');

# 해당일자의 년도 리턴
SELECT year(curdate());
# 해당일자의 월 리턴
SELECT month(curdate());
# 해당일자의 일 리턴
SELECT day(curdate());
# 해당일자의 요일 리턴
SELECT weekday(curdate());  # 월:0, 화:1, ... 일:7
SELECT dayofweek(curdate()); # 일:1, 월:2, 화:3, ...토:7
# 일년 중 몇번째 주인지 리턴
SELECT week(curdate());


