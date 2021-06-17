<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>html_018_process.jsp</title>
</head>
<body>
 <%
   request.setCharacterEncoding("UTF-8");
   String other=request.getParameter("other");
 %>
 
 <p><%=other%> </p>
</body>
</html>






