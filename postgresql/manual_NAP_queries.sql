select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category = 10
        and abs(m.refdist1 - p.refdist1) < (1609.34 * 1.0)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * 1.0)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * 1.0)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 1.0)
        limit 1
    ) t2 on true
    where m.category = 14;

--  mcount | tcount 
-- --------+--------
--   10000 |   3627
-- (1 row)
-- 36.27%
-- Time: 85394.043 ms (01:25.394)


select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category = 11
        and abs(m.refdist1 - p.refdist1) < (1609.34 * 1.0)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * 1.0)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * 1.0)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 1.0)
        limit 1
    ) t2 on true
    where m.category = 14;

--  mcount | tcount 
-- --------+--------
--   10000 |   7081
-- (1 row)
-- 70.81%
-- Time: 41213.537 ms (00:41.214)


select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category = 12
        and abs(m.refdist1 - p.refdist1) < (1609.34 * 1.0)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * 1.0)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * 1.0)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 1.0)
        limit 1
    ) t2 on true
    where m.category = 14;

--  mcount | tcount 
-- --------+--------
--   10000 |   7637
-- (1 row)
-- 76.375
-- Time: 35532.944 ms (00:35.533)


select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category = 13
        and abs(m.refdist1 - p.refdist1) < (1609.34 * 1.0)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * 1.0)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * 1.0)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 1.0)
        limit 1
    ) t2 on true
    where m.category = 14;

--  mcount | tcount 
-- --------+--------
--   10000 |   8366
-- (1 row)
-- 83.66%
-- Time: 24689.992 ms (00:24.690)


select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category = 14
        and abs(m.refdist1 - p.refdist1) < (1609.34 * 1.0)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * 1.0)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * 1.0)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 1.0)
        limit 1
    ) t2 on true
    where m.category = 14;

--  mcount | tcount 
-- --------+--------
--   10000 |  10000
-- (1 row)
-- 100.00%
-- Time: 7816.878 ms (00:07.817)


select count(m.sampleid) as mcount,
        count(t2.sampleid) as tcount
    from sample_cat_ref_dists m
    left join lateral (select p.sampleid from sample_cat_ref_dists p
        where
        p.category = 15
        and abs(m.refdist1 - p.refdist1) < (1609.34 * 1.0)
        and abs(m.refdist2 - p.refdist2) < (1609.34 * 1.0)
        and abs(m.refdist3 - p.refdist3) < (1609.34 * 1.0)
        and st_distance(p.st_geompoint, m.st_geompoint) <= (1609.34 * 1.0)
        limit 1
    ) t2 on true
    where m.category = 14;

--  mcount | tcount 
-- --------+--------
--   10000 |   9350
-- (1 row)
-- 93.50%
-- Time: 25410.502 ms (00:25.411)

