# 아래 두 식은 동일하가 FROM 절 생략 가
SELECT 1+2;
SELECT 1+2 FROM dual;

# 데이터 검색 : SELECT 
# SELECT 절
# FROM 절
# WHERE 절
# group by 절
# Having 절
# ORDER by 절

# [해석순서]
# FROM 절
# WHERE 절
# group by 절
# Having 절
# SELECT 절
# ORDER by 절


SELECT customerNumber, customerName, phone, city
FROM customers;

# 모든 테이블에 대한 정보를 가져온다. * 는 모두의 의미
# 고객 전체의 정보를 검색하시오
SELECT * FROM customers;


# database 변경
use classicmodels

# customers 데이터 안에 있는 city 데이터 가져오기
SELECT city FROM customers;

# distinct 로 중복 제거하기
SELECT distinct city form customers;

# city에 있는 레코드의 수 가져오기
SELECT count(city) FROM customers;

SELECT count(customerNumber), count(city) FROM customers;

SELECT count(customerNumber), city
FROM customers
group by city;

# 도시가 Nantes 인 것만 가져오기
SELECT sum(CASE WHEN city='Nantes' THEN 1 ELSE 0 END) cnt
FROM customers;

# 다중 조건문
SELECT sum(CASE WHEN city='Nantes' THEN 1
			    WHEN city='Las Vegas' THEN 2
                WHEN city='Stavern' THEN 3 ELSE 0 END) cnt
FROM customers;

SELECT city, CASE WHEN city='Nantes' THEN 1
			    WHEN city='Las Vegas' THEN 2
                WHEN city='Stavern' THEN 3 ELSE 0 END cnt
FROM customers;


# 교재 46페이지
SELECT SUM(CASE WHEN country='USA' THEN 1 ELSE 0 END) AS N_USA,
       SUM(CASE WHEN country='USA' THEN 1 ELSE 0 END)/COUNT(*) AS USA_PORTION
FROM customers;       

# count(*) : 무조건 전체 레코드 수를 리턴한다.
# count(컬럼명) : NULL이 아닌 전체 레코드 수를 리턴한다.alter
SELECT count(employeeNumber), count(reportsTO), count(*)
FROM employees;


#___

# JOIN
SELECT * FROM orders;
SELECT * FROM customers;
# 두개의 테이블에 같은 종류의 데이터가 저장되어있는지 살펴본다.

# orders 테이블과 customers 테이블에 customerNumber 데이터가 동일하다.
# 조인함수로 두개 테이블 결합하기
SELECT * FROM orders LEFT JOIN customers
ON orders.customerNumber = customers.customerNumber;

# LEFT JOIN (A 테이블은 모두, B는 공통적인 것만 가지고 오겠다)
SELECT o.orderNumber, o.orderNumber,  c.customerNumber
FROM orders o LEFT JOIN customers c
ON o.customerNUmber = c.customerNumber;

# INNER JOIN (두개 테이블에 공통적인 것만 가져옴)
SELECT o.orderNumber, o.orderNumber,  c.customerNumber
FROM orders o INNER JOIN customers c
ON o.customerNUmber = c.customerNumber;

# LEFT JOIN 과 INNER JOIN 갯수가 같음 즉 o.customerNumber과 c.customerNumber 가 일치함

# p.56 페이지 북미, 비북미 출력하기
# 나라가 USA, Canada 면 North America로 출력, 아니면 OTHERS로 출력
SELECT country,
CASE WHEN country in ('USA', 'Canada') then 'North America' ELSE 'OTHERS' END AS region
FROM classicmodels.customers

# p.57 classicmodels.crstomers의 country 칼럼을 이용해 북미, 비북미를 출력하는 칼럼을 생성하고, 북미, 비북미 거주 고객의 수를 계산하세요.
SELECT CASE WHEN country IN ('USA', 'Canada') then 'North America'
ELSE 'OTHERS' END AS region,
COUNT(customerNumber) N_customers
FROM classicmodels.customers
GROUP
BY CASE WHEN country IN ('USA', 'Canada') then 'North America'
ELSE 'OTHERS' END