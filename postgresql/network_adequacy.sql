
-- Original SQL Server version:
-- select MEMBER_ID
-- from MemberTable m
-- cross apply (select top(1) PROVIDER_ID from ProviderTable p with(index(geopoint_index))
--        where
--            abs(m.distance_1 - p.distance_1) &amp;amp;amp;lt; 10
--        and abs(m.distance_2 - p.distance_2) &amp;amp;amp;lt; 10
--        and abs(m.distance_3 - p.distance_3) &amp;amp;amp;lt; 10
--        and p.geopoint.STDistance(m.geopoint) &amp;amp;amp;lt;= (1609.34 * 10)
--        and p.ProviderType = 'PCP'


select count(m.sampleid) as mcount,
       count(t2.sampleid) as tcount
from sample_cat_ref_dists m
left join lateral (select p.sampleid from sample_cat_ref_dists p
       where
       p.category = 11
       and abs(m.refdist1 - p.refdist1) < (1609.34 * 10)
       and abs(m.refdist2 - p.refdist2) < (1609.34 * 10)
       and abs(m.refdist3 - p.refdist3) < (1609.34 * 10)
       and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 10)
       limit 1
) t2 on true
where m.category = 10;

select count(m.sampleid) as mcount,
       count(t2.sampleid) as tcount
from sample_cat_ref_dists m
left join lateral (select p.sampleid from sample_cat_ref_dists p
       where
       p.category = 11
       and abs(m.refdist1 - p.refdist1) < (1609.34 * 10)
       and abs(m.refdist2 - p.refdist2) < (1609.34 * 10)
       and abs(m.refdist3 - p.refdist3) < (1609.34 * 10)
       limit 1
) t2 on true
where m.category = 10;


select m.sampleid, t1.sampleid
from sample_cat_ref_dists m
inner join lateral (
    select p1.sampleid
    from sample_cat_ref_dists p1
    join sample_cat_ref_dists p2 on
        p1.sampleid = p2.sampleid
    join sample_cat_ref_dists p3 on
        p1.sampleid = p3.sampleid and
        p2.sampleid = p3.sampleid
    where
        p1.category = 11
        and abs(m.refdist1 - p1.refdist1) < (1609.34 * 10)
        and abs(m.refdist1 - p2.refdist1) < (1609.34 * 10)
        and abs(m.refdist1 - p3.refdist1) < (1609.34 * 10)
        limit 1
) t1 on true
where m.category = 10;

select st_distance(a.st_geompoint, b.st_geompoint) as dist
from sample_cat_ref_dists a
join sample_cat_ref_dists b
on a.sampleid = 66 and b.sampleid = 101;



select m.sampleid mid,
       t2.sampleid tid,
       t2.bubba
from sample_cat_ref_dists m
left join lateral (select p.sampleid,
            st_distance(p.st_geompoint, m.st_geompoint) as bubba
       from sample_cat_ref_dists p
       where
       p.category = 11
       and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 10)
       limit 1
) t2 on true
where m.category = 10;



select count(m.sampleid) as mcount,
       count(t2.sampleid) as tcount,
       min(t2.bubba), max(t2.bubba)
from sample_cat_ref_dists m
left join lateral (select p.sampleid,
            st_distance(p.st_geompoint, m.st_geompoint) as bubba
       from sample_cat_ref_dists p
       where
       p.category = 11
       and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 0.1)
       limit 1
) t2 on true
where m.category = 10;


select m.sampleid, st_distance(p.st_geompoint, m.st_geompoint)
from sample_cat_ref_dists m
join sample_cat_ref_dists p on
    m.sampleid = p.sampleid
where m.category = 10
    and p.category = 11
order by st_distance(p.st_geompoint, m.st_geompoint);


create or replace function funcs.CatTrilatAdequacy(leftcat int, rightcat int, mydist float,
            notes varchar(1000) default 'adequacy_trilat') RETURNS numeric(18,3)
AS $$
   declare StartTime timestamp;
           EndTime timestamp;
           elapsedms numeric(18,3);
BEGIN
   -- No, we do ZERO security testing on inputs.  Carry on.
   StartTime := clock_timestamp();

    perform count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category = leftcat
        and abs(m.refdist1 - p.refdist1) < (1609.34 * mydist)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * mydist)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * mydist)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * mydist)
        limit 1
    ) t2 on true
    where m.category = rightcat;

   EndTime := clock_timestamp();
   elapsedms := cast(extract(epoch from (EndTime - StartTime)) as numeric(18,3));
   insert into query_timings (leftcat, rightcat, timing_ms, notes) values (leftcat, rightcat, elapsedms, notes);
   RETURN elapsedms;
END
$$  LANGUAGE plpgsql
;


create or replace function funcs.CatTrilatApproxAdequacy_2(leftcat int, rightcat int, mydist float,
            notes varchar(1000) default 'adequacy_approx_2_trilat') RETURNS numeric(18,3)
AS $$
   declare StartTime timestamp;
           EndTime timestamp;
           elapsedms numeric(18,3);
BEGIN
   -- No, we do ZERO security testing on inputs.  Carry on.
   StartTime := clock_timestamp();

    perform count(m.sampleid)
    from sample_cat_ref_dists m
    inner join lateral (
        select p.sampleid from sample_cat_ref_dists p
            where
            p.category = leftcat
            and abs(m.refdist1 - p.refdist1) < (1609.34 * mydist)
            limit 1
    ) t1 on true
    inner join lateral (
        select p.sampleid from sample_cat_ref_dists p
            where
            p.category = leftcat
            and abs(m.refdist2 - p.refdist2) < (1609.34 * mydist)
            limit 1
    ) t2 on true
    inner join lateral (
        select p.sampleid from sample_cat_ref_dists p
            where
            p.category = leftcat
            and abs(m.refdist2 - p.refdist2) < (1609.34 * mydist)
            limit 1
    ) t3 on true
    where m.category = rightcat;

   EndTime := clock_timestamp();
   elapsedms := cast(extract(epoch from (EndTime - StartTime)) as numeric(18,3));
   insert into query_timings (leftcat, rightcat, timing_ms, notes) values (leftcat, rightcat, elapsedms, notes);
   RETURN elapsedms;
END
$$  LANGUAGE plpgsql
;



create or replace function funcs.CatTrilatApproxAdequacy(leftcat int, rightcat int, mydist float,
            notes varchar(1000) default 'adequacy_approx_trilat') RETURNS numeric(18,3)
AS $$
   declare StartTime timestamp;
           EndTime timestamp;
           elapsedms numeric(18,3);
BEGIN
   -- No, we do ZERO security testing on inputs.  Carry on.
   StartTime := clock_timestamp();

    perform count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category = leftcat
        and abs(m.refdist1 - p.refdist1) < (1609.34 * mydist)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * mydist)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * mydist)
        limit 1
    ) t2 on true
    where m.category = rightcat;

   EndTime := clock_timestamp();
   elapsedms := cast(extract(epoch from (EndTime - StartTime)) as numeric(18,3));
   insert into query_timings (leftcat, rightcat, timing_ms, notes) values (leftcat, rightcat, elapsedms, notes);
   RETURN elapsedms;
END
$$  LANGUAGE plpgsql
;


create or replace function funcs.CatAdequacy(leftcat int, rightcat int, mydist float,
            notes varchar(1000) default 'adequacy_trilat') RETURNS numeric(18,3)
AS $$
   declare StartTime timestamp;
           EndTime timestamp;
           elapsedms numeric(18,3);
BEGIN
   -- No, we do ZERO security testing on inputs.  Carry on.
   StartTime := clock_timestamp();

    perform count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category = leftcat
        -- and abs(m.refdist1 - p.refdist1) < mydist
        -- and abs(m.refdist2 - p.refdist2) < mydist
        -- and abs(m.refdist3 - p.refdist3) < mydist
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * mydist)
        limit 1
    ) t2 on true
    where m.category = rightcat;

   EndTime := clock_timestamp();
   elapsedms := cast(extract(epoch from (EndTime - StartTime)) as numeric(18,3));
   insert into query_timings (leftcat, rightcat, timing_ms, notes) values (leftcat, rightcat, elapsedms, notes);
   RETURN elapsedms;
END
$$  LANGUAGE plpgsql
;


create or replace function funcs.RunAdequacyTimings(notes varchar(1000) default 'Adequacy Timings: ',
       min_cat int default 10, max_cat int default 11, dist float default 10) RETURNS int
as $$
   -- Really, no input sanitization or robustness.  It's ok.
  declare i int;  j int;
  declare tnote varchar(1000);
BEGIN
   notes = notes || ' - dist: ' || dist;
   for i in min_cat..max_cat LOOP
      for j in min_cat..max_cat LOOP
         tnote = 'CatAdequacy: ' || notes;
         perform funcs.CatAdequacy(i, j, dist, tnote);
         tnote = 'CatTrilatAdequacy: ' || notes;
         perform funcs.CatTrilatAdequacy(i, j, dist, tnote);
        --  tnote = 'CatTrilatApproxAdequacy: ' || notes;
        --  perform funcs.CatTrilatApproxAdequacy(i, j, dist, tnote);
        --  tnote = 'CatTrilatApproxAdequacy_2: ' || notes;
        --  perform funcs.CatTrilatApproxAdequacy_2(i, j, dist, tnote);
      END LOOP;
   END LOOP;
   RETURN i;
END
$$ LANGUAGE plpgsql;


select funcs.RunAdequacyTimings(notes := 'Adequacy Timings: ', min_cat := 10, max_cat := 15, dist := 0.1);
select funcs.RunAdequacyTimings(notes := 'Adequacy Timings: ', min_cat := 10, max_cat := 15, dist := 1);
select funcs.RunAdequacyTimings(notes := 'Adequacy Timings: ', min_cat := 10, max_cat := 15, dist := 10);
select funcs.RunAdequacyTimings(notes := 'Adequacy Timings: ', min_cat := 10, max_cat := 15, dist := 100);

select * from v_results;

select count(1), avg(timing_ms), notes
from v_results
group by notes;
