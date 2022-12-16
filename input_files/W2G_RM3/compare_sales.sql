
/* compare_sales.sql sample SQL script

	Run this script on the tutorial.mdb database using RUNSQLSCRIPT.
 
    Copyright 1999-2011 The MathWorks, Inc.
       

*/


-- Query to get sales of products from U.S suppliers in the first quarter

select      productDescription, supplierName, city, January as Jan_Sales, February as Feb_Sales, March as Mar_Sales
from 		suppliers A,
			productTable B,
			salesVolume C
where		A.Country		like 	'United States'
AND         A.SupplierNumber	=	B.SupplierNumber
AND         B.stocknumber		= 	C.stockNumber
;


-- Query to get sales of products from foreign suppliers in the first quarter

select      productDescription, supplierName, city, January as Jan_Sales, February as Feb_Sales, March as Mar_Sales
from 		suppliers A,
			productTable B,
			salesVolume C
where		A.Country	not like		'United States'
AND         A.SupplierNumber	= 		B.SupplierNumber
AND         B.stocknumber		= 		C.stockNumber
;



