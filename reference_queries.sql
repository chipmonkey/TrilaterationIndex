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

create table query_timings (leftcat int, rightcat int, timing_ms numeric(18,3));
alter table query_timings add column id serial primary key;
alter table query_timings add column notes varchar(1000);
alter table query_timings add column row_stamp timestamp default now();

create or replace function CatMinDist_timing(leftcat int, rightcat int, notes varchar(1000) default 'no notes') RETURNS numeric(18,3)
AS $$
   declare StartTime timestamp;
           EndTime timestamp;
           elapsedms numeric(18,3);
BEGIN
   -- No, we do ZERO security testing on inputs.  Carry on.
   StartTime := clock_timestamp();
   perform a.sampleid,
     min(st_distance(a.st_geompoint, b.st_geompoint)) as min_st_distance
   from sample_cat_gis a, sample_cat_gis b
   where a.category = leftcat and b.category = rightcat
   group by a.sampleid;
   EndTime := clock_timestamp();
   elapsedms := cast(extract(epoch from (EndTime - StartTime)) as numeric(18,3));
   insert into query_timings (leftcat, rightcat, timing_seconds, notes) values (leftcat, rightcat, elapsedms, notes);
   RETURN elapsedms;
END
$$  LANGUAGE plpgsql
;

-- insert into query_timings select 9 as leftcat, 10 as rightcat, catmindist_timing(9, 10) as timing_ms;
select 9 as leftcat, 10 as rightcat, catmindist_timing(9, 10, 'Initial Tests') as timing_ms;

create index sample_cat_gix on sample_cat_gis using gist(st_geompoint);

create or replace function Run_timings(notes varchar(1000) default 'run_timings') RETURNS int
as $$
   -- Really, no input sanitization or robustness.  It's ok.
  declare i int;  j int;
BEGIN
   for i in 0..15 LOOP
      for j in 0..15 LOOP
         perform catmindist_timing(i, j, notes);
      END LOOP;
   END LOOP;
   RETURN i;
END
$$ LANGUAGE plpgsql;


