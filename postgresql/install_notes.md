sudo apt-get install postgresql-12

# OR run postgres docker and connect as root -see runme.sh
apt install postgis --no-install-recommends
apt install postgresql-12-postgis-3-scripts

create database trilat;
\c trilat
create extension postgis;
create user chipmonkey with superuser;
grant all privileges on database trilat to chipmonkey;
\c trilat
\q

### For reference, here are the categories:
```
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
(16 rows)
```


# NO PSQL Index:
Some postgres things:
`time psql -qtAc " <big query> ";
`EXPLAIN (ANALYZE ON, BUFFERS ON)`
## for 1 Query Point and 10,000 data points
via: https://postgis.net/workshops/postgis-intro/knn.html
```
SELECT
  a.sampleid, b.sampleid,
  ST_Distance(a.st_geompoint, b.st_geompoint)
FROM
  sample_cat_gis a,
  sample_cat_gis b
WHERE
  a.sampleid = 1 and
  b.category = 5
ORDER BY ST_Distance(a.st_geompoint, b.st_geompoint) ASC
LIMIT 5;
```

 sampleid | sampleid 
----------+----------
        1 |   103625
        1 |   125715
        1 |    77125
        1 |    55485
        1 |    89055
(5 rows)

Time: 212.708 ms

Subsequent attempts:
2. Time: 73.251 ms
3. Time: 80.553 ms
4. Time: 54.169 ms
5. Time: 54.382 ms


## Select 5-NN for 100 Query points and 10,000 data points
select a.sampleid as QSampleID, b.sampleid as NNSampleID
from
  sample_cat_gis as a
cross join lateral (
    select sampleid, c.st_geompoint
    from sample_cat_gis c
    where
    c.category = 5
    order by ST_Distance(a.st_geompoint, c.st_geompoint)
    limit 5
    ) as b
where
    a.category = 10;

This took ~25 seconds.  Subsequent runs were almost identical.
Time: 2579.389 ms (00:02.579)

So a 10x increse in query time for 100x more rows

### FWIW, this uses only 1 CPU

If you do something insane, like 10,000 x 100,000 you timeout:

ERROR:  57014: canceling statement due to user request
LOCATION:  ProcessInterrupts, postgres.c:3124
Time: 508650.669 ms (08:28.651)

## Select 5-NN for 100 x 25,000
`c.category in (1, 2, 13)`

Results:
Time: 5361.105 ms (00:05.361)

## Select 20-NN for 100 x 10,000
Time: 5922.942 ms (00:05.923)

## Select 5-NN for 100 x 50,000
Time: 10533.188 ms (00:10.533)

## Select 5-NN for 100 x 150,000
Time: 28124.634 ms (00:28.125)

# Create Index
CREATE INDEX idx_gist_geography ON sample_cat_gis USING gist(st_geompoint);
CREATE INDEX
Time: 751.123 ms

# Timings with Index:
Top|Q|D
---
5|1|10,000
Time: 64.694 ms

5|100|10,000
Time: 2568.415 ms (00:02.568)

5|100|25,000
Time: 5384.597 ms (00:05.385)

20|100|10,000
Time: 4316.469 ms (00:04.316)

5|100|50,000
Time: 10048.796 ms (00:10.049)

5|100|150,000
Time: 27850.183 ms (00:27.850)

# Those aren't much different

##  We need a python program for this...
