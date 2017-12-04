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
copy sample_categorized from '~/Documents/GradSchool/Thesis/TrilaterationIndex/data/lat_long_categorized.csv' with (FORMAT csv);

select SampleID, ST_MakePoint(Longitude, Latitude) as st_geompoint, Longitude, Latitude, category into sample_cat_gis from sample_categorized;

create table referencepoints (Name varchar(50), Latitude numeric, Longitude numeric, st_refpoint geometry;


insert into referencepoints (name, latitude, longitude) values ('North Pole', 90, 0);
insert into referencepoints (name, latitude, longitude) values ('Louisville KY', 38.26, -85.76);
insert into referencepoints (name, latitude, longitude) values ('New Caledonia', -19.22, 159.93);
insert into referencepoints (name, latitude, longitude) values ('Aldabra', -9.42, 46.63);
update referencepoints set st_refpoint = st_makepoint(Latitude, Longitude);

alter table sample_categorized add column SampleID serial unique;
alter table referencepoints add column RefID serial unique;

