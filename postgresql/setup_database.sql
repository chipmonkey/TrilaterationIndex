-- Destructive (for known tables) - initiates sample data in the database
-- The only table NOT destroyed is query_timings, if it exists
-- This is pretty specific to the task at hand

\timing on
\pset pager 0


-- Load sample latitudes and longitudes
-- Table: sample_categorized
drop table if exists sample_categorized;
create table sample_categorized (Latitude numeric, Longitude numeric, Category smallint);
-- COPY sample_categorized from '/home/chipmonkey/Documents/GradSchool/Thesis/TrilaterationIndex/data/lat_long_categorized.csv' CSV HEADER; 
COPY sample_categorized from '/input/lat_long_categorized.csv' CSV HEADER;
alter table sample_categorized add column SampleID serial unique;


-- Create copy of lat/long using PostGIS geometry
-- Table: sample_cat_gis
drop table if exists sample_cat_gis;
select SampleID, ST_SetSRID(ST_Point(Longitude, Latitude), 4326)::geography as st_geompoint, 
Longitude, Latitude, category into sample_cat_gis from sample_categorized;


-- Create table with default reference points
-- Table: reference_points
drop table if exists reference_points;
create table reference_points (Name varchar(50), Latitude numeric, Longitude numeric, st_refpoint geography);
insert into reference_points (name, latitude, longitude) values ('North Pole', 90, 0);
insert into reference_points (name, latitude, longitude) values ('Louisville KY', 38.26, -85.76);
insert into reference_points (name, latitude, longitude) values ('Phantom Sandy Island', -19.22, 159.93);
insert into reference_points (name, latitude, longitude) values ('Aldabra', -9.42, 46.63);
update reference_points set st_refpoint = st_makepoint(Latitude, Longitude);

alter table reference_points add column RefID serial unique;


-- Create distances between data and ref points
-- This is essentially the 'fit' or build time for the index

-- a PIVOT or CROSSTAB would be more elegant but
-- SELECT 150000 - Time: 1434.859 ms (00:01.435)
drop table if exists sample_cat_ref_dists;
select scg.sampleid, scg.st_geompoint, scg.longitude, scg.latitude, scg.category,
  st_distance(scg.st_geompoint, rp1.st_refpoint) as RefDist1,
  st_distance(scg.st_geompoint, rp2.st_refpoint) as RefDist2,
  st_distance(scg.st_geompoint, rp3.st_refpoint) as RefDist3,
  st_distance(scg.st_geompoint, rp4.st_refpoint) as RefDist4
into sample_cat_ref_dists from sample_cat_gis scg
join reference_points rp1 on rp1.refid = 1
join reference_points rp2 on rp2.refid = 2
join reference_points rp3 on rp3.refid = 3
join reference_points rp4 on rp4.refid = 4;

create index sample_cat_ref_dists_ref1 on sample_cat_ref_dists (RefDist1);
create index sample_cat_ref_dists_ref2 on sample_cat_ref_dists (RefDist2);
create index sample_cat_ref_dists_ref3 on sample_cat_ref_dists (RefDist3);
create index sample_cat_ref_dists_ref4 on sample_cat_ref_dists (RefDist4);


-- Test query just to make sure some things work:
-- via: https://postgis.net/workshops/postgis-intro/knn.html
EXPLAIN (ANALYZE ON, BUFFERS ON)
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



-- The above sets up the initial data; but none of the monkey indexes


-- Functions for query testing:
create table if not exists query_timings (leftcat int, rightcat int, timing_ms numeric(18,3));
alter table query_timings add column if not exists id serial primary key;
alter table query_timings add column if not exists notes varchar(1000);
alter table query_timings add column if not exists row_stamp timestamp default now();

create schema if not exists funcs;

create or replace function funcs.CatMinDist_timing(leftcat int, rightcat int, notes varchar(1000) default 'no notes') RETURNS numeric(18,3)
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
   insert into query_timings (leftcat, rightcat, timing_ms, notes) values (leftcat, rightcat, elapsedms, notes);
   RETURN elapsedms;
END
$$  LANGUAGE plpgsql
;


create or replace function funcs.CatMinDist_sq_timing(leftcat int, rightcat int, notes varchar(1000) default 'sqrt approach') RETURNS numeric(18,3)
  AS $$
     declare StartTime timestamp;
             EndTime timestamp;
             elapsedms numeric(18,3);
  BEGIN
     -- No, we do ZERO security testing on inputs.  Carry on.
     StartTime := clock_timestamp();
     perform a.sampleid,
       min(sqrt(abs(a.refdist1 - b.refdist1)^2 + abs(a.refdist2-b.refdist2)^2 + abs(a.refdist3-b.refdist3)^2)) as min_st_distance
     from sample_cat_ref_dists a, sample_cat_ref_dists b
     where a.category = leftcat and b.category = rightcat
     group by a.sampleid;
     EndTime := clock_timestamp();
     elapsedms := cast(extract(epoch from (EndTime - StartTime)) as numeric(18,3));
     insert into query_timings (leftcat, rightcat, timing_ms, notes) values (leftcat, rightcat, elapsedms, notes);
     RETURN elapsedms;
  END
  $$  LANGUAGE plpgsql
  ;


-- select
--     a.sampleid,
--     min(sqrt(abs(a.refdist1 - b.refdist1)^2 + abs(a.refdist2-b.refdist2)^2 + abs(a.refdist3-b.refdist3)^2)) as min_st_distance
-- from sample_cat_ref_dists a, sample_cat_ref_dists b
-- where a.category = 10 and b.category = 11
-- group by a.sampleid


-- insert into query_timings select 9 as leftcat, 10 as rightcat, CatMinDist_timing(9, 10) as timing_ms;
select 9 as leftcat, 10 as rightcat, funcs.CatMinDist_timing(9, 10, 'Initial Tests') as timing_ms;

create index sample_cat_gix on sample_cat_gis using gist(st_geompoint);

create or replace function funcs.Run_timings(notes varchar(1000) default 'min_stdist_timings',
       min_cat int default 10, max_cat int default 13) RETURNS int
as $$
   -- Really, no input sanitization or robustness.  It's ok.
  declare i int;  j int;
BEGIN
   for i in min_cat..max_cat LOOP
      for j in min_cat..max_cat LOOP
         perform funcs.CatMinDist_timing(i, j, notes);
      END LOOP;
   END LOOP;
   RETURN i;
END
$$ LANGUAGE plpgsql;


create or replace function funcs.Run_sq_timings(notes varchar(1000) default 'approx_trilat_timings',
       min_cat int default 10, max_cat int default 13) RETURNS int
as $$
   -- Really, no input sanitization or robustness.  It's ok.
  declare i int;  j int;
BEGIN
   for i in min_cat..max_cat LOOP
      for j in min_cat..max_cat LOOP
         perform funcs.CatMinDist_sq_timing(i, j, notes);
      END LOOP;
   END LOOP;
   RETURN i;
END
$$ LANGUAGE plpgsql;
 
select funcs.run_timings();
select funcs.run_sq_timings();
