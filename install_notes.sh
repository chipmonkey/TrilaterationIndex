# postgis wants postgresql 10.0 but default is 9.6:
sudo apt-get install postgresql-9.6
sudo apt install postgis --no-install-recommends
sudo apt-get install postgresql-9.6-postgis-scripts

sudo -u postgres psql
create extension postgis;
create database trilateration;
create user chipmonkey with superuser;
grant all privileges on database trilateration to chipmonkey;
\c trilateration
\q

create table sample_categorized (Latitude numeric, Longitude numeric, Category smallint);
COPY sample_categorized from '/home/chipmonkey/Documents/GradSchool/Thesis/TrilaterationIndex/data/lat_long_categorized.csv' CSV HEADER; 
alter table sample_categorized add column SampleID serial unique;

select SampleID, ST_SetSRID(ST_Point(Longitude, Latitude), 4326)::geography as st_geompoint, 
Longitude, Latitude, category into sample_cat_gis from sample_categorized;

create table referencepoints (Name varchar(50), Latitude numeric, Longitude numeric, st_refpoint geography);


insert into referencepoints (name, latitude, longitude) values ('North Pole', 90, 0);
insert into referencepoints (name, latitude, longitude) values ('Louisville KY', 38.26, -85.76);
insert into referencepoints (name, latitude, longitude) values ('Phantom Sandy Island', -19.22, 159.93);
insert into referencepoints (name, latitude, longitude) values ('Aldabra', -9.42, 46.63);
update referencepoints set st_refpoint = st_makepoint(Latitude, Longitude);

alter table referencepoints add column RefID serial unique;

select SampleID, RefID, st_distance(s.st_geompoint, r.st_refpoint)
into sample_ref_distances from sample_cat_gis s cross join referencepoints r;
