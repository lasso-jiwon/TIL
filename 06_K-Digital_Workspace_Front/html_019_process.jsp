<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>html_019_process.jsp</title>
</head>
<body>


<%
  String[] subject=request.getParameterValues("subject");
  if(subject!=null){
    for(int i=0; i<subject.length; i++){ 
	  
%>
 <p><%=subject[i] %> </p>

<%
    }
  }
  
 String fruit=request.getParameter("fruit"); 
%>

<p><%=fruit %></p>
</body>
</html>





