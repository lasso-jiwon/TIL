<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>html_016_process.jsp</title>
</head>
<body>
<%
    //?fid=min&fpass=1234
   String fid=request.getParameter("fid");
   String fpass=request.getParameter("fpass");
%>

<p><%=fid %> / <%=fpass %></p>
</body>
</html>









