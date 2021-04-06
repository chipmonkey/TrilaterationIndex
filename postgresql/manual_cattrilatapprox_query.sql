select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category in (10, 11, 12, 13, 14, 15)
        and abs(m.refdist1 - p.refdist1) < (1609.34 * 0.1)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * 0.1)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * 0.1)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 0.1)
        limit 1
    ) t2 on true
    where m.category in (0, 1, 2, 3, 4, 5);

532.328953

insert into query_timings (leftcat, rightcat, timing_ms, notes, row_stamp)
values (1015, 012345, 532.328953, 'Manual CatTrilatAdequacy cats 10..15, cats 0..5 - dist: 0.1', now())


select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category in (0, 1, 2, 3, 4, 5)
        and abs(m.refdist1 - p.refdist1) < (1609.34 * 0.1)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * 0.1)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * 0.1)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 0.1)
        limit 1
    ) t2 on true
    where m.category in (10, 11, 12, 13, 14, 15);

insert into query_timings (leftcat, rightcat, timing_ms, notes, row_stamp)
values (012345, 1015, 789.871561, 'Manual CatTrilatAdequacy cats 0..5, cats 10..15 - dist: 0.1', now())

789871.561 ms


select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category in (0, 1, 2, 3, 4, 5)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 0.1)
        limit 1
    ) t2 on true
    where m.category in (10, 11, 12, 13, 14, 15);

select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_gis m
    left join lateral (select p.sampleid from sample_cat_gis p
        where
        p.category in (0, 1, 2, 3, 4, 5)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 0.1)
        limit 1
    ) t2 on true
    where m.category in (10, 11, 12, 13, 14, 15);


SELECT EXTRACT(EPOCH FROM timestamptz '2013-07-01 12:00:00') -
       EXTRACT(EPOCH FROM timestamptz '2013-03-01 12:00:00');

SELECT EXTRACT(EPOCH FROM timestamptz '2013-07-20 12:00:07' -
       	 	          timestamptz '2013-07-20 12:00:00');
