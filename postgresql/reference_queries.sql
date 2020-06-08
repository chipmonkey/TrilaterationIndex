select count(1), category 
from sample_cat_gis 
group by category
order by category;

count | category 
-------+----------
 10000 |        0
 10000 |        1
 10000 |        2
 10000 |        3
 10000 |        4
 10000 |        5
 10000 |        6
 10000 |        7
 10000 |        8
 10000 |        9
   100 |       10
   900 |       11
  2000 |       12
  5000 |       13
 10000 |       14
 32000 |       15


\timing

select a.sampleid, 
     min(st_distance(a.st_geompoint, b.st_geompoint)) as min_st_distance
from sample_cat_gis a, sample_cat_gis b
where a.category = 10 and b.category = 11
group by a.sampleid;
-- 258ms



copy query_timings to '/tmp/query_timings.csv' DELIMITER ',' CSV HEADER;
COPY (SELECT * FROM v_category_counts) to '/tmp/category_counts.csv' DELIMITER ',' CSV HEADER;


trilateration=# select sum(timing_seconds), notes
from query_timings
group by notes;
    sum    |            notes             
-----------+------------------------------
     0.097 | test
 54866.090 | Full cycle default gis index
 13758.834 | Initial sqrt run
   146.951 | GIS Index
    29.058 | Vacuumed
   942.526 | Initial Tests
(6 rows)
