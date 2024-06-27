create type detection_type AS ENUM ('background', 'person', 'car', 'bicycle');

create type detection_class AS ENUM ('Unknown',   'Compacts',   'Sedans',   'SUVs',   'Coupes',   'Muscle',   'SportsClassics',   'Sports',   'Super',   'Motorcycles',   'OffRoad',   'Industrial',   'Utility',   'Vans',   'Cycles',   'Boats',   'Helicopters',   'Planes',   'Service',   'Emergency',   'Military',   'Commercial',   'Trains');

create type weather AS ENUM ('Unknown', 'ExtraSunny', 'Clear', 'Clouds', 'Smog', 'Foggy', 'Overcast', 'Raining', 'ThunderStorm', 'Clearing', 'Neutral', 'Snowing', 'Blizzard', 'Snowlight', 'Christmas', 'Halloween');

create table detections
(
	detection_id serial not null
		constraint detections_pkey
			primary key,
	snapshot_id integer,
	type detection_type,
	pos geometry(PointZ),
	bbox box,
	class detection_class default 'Unknown'::detection_class,
	handle integer default '-1'::integer,
	best_bbox box,
	best_bbox_old box,
	bbox3d box3d,
	rot geometry,
	coverage real default 0.0,
    created timestamp without time zone default (now() at time zone 'utc')
)
;


create table runs
(
	run_id serial not null
		constraint runs_pkey
			primary key,
	runguid uuid,
	archivepath text,
	localpath text,
	session_id integer default 1,
	instance_id integer default 0,
    created timestamp without time zone default (now() at time zone 'utc')
)
;


create table sessions
(
	session_id serial not null
		constraint sessions_pkey
			primary key,
	name text
		constraint sessions_name_key
			unique,
	start timestamp with time zone,
	"end" timestamp with time zone,
    created timestamp without time zone default (now() at time zone 'utc')
)
;

alter table runs
	add constraint runs_session_fkey
		foreign key (session_id) references sessions
			on delete cascade
;

create table snapshots
(
	snapshot_id serial not null
		constraint snapshots_pkey
			primary key,
	run_id integer
		constraint snapshots_run_fkey
			references runs
				on delete cascade,
	version integer,
	imagepath text,
	timestamp timestamp with time zone,
	timeofday time,
	currentweather weather,
	camera_pos geometry(PointZ),
	camera_direction geometry,
	camera_fov real,
	view_matrix double precision[],
	proj_matrix double precision[],
	processed boolean default false not null,
	width integer,
	height integer
)
;


alter table detections
	add constraint detections_snapshot_fkey
		foreign key (snapshot_id) references snapshots
			on delete cascade
;

create table instances
(
	instance_id serial not null
		constraint isntances_pkey
			primary key,
	hostname text,
	instanceid text
		constraint instanceid_uniq
			unique,
	instancetype text,
	publichostname text,
	amiid text,
    created timestamp without time zone default (now() at time zone 'utc'),
	constraint instance_info_uniq
		unique (hostname, instanceid, instancetype, publichostname, amiid)
)
;

alter table runs
	add constraint runs_instance_fkey
		foreign key (instance_id) references instances
;

create table snapshot_weathers
(
	weather_id serial not null
		constraint snapshot_weathers_pkey
			primary key,
	snapshot_id integer
		constraint snapshot_weathers_snapshot_id_fkey
			references snapshots
				on delete cascade,
	weather_type weather,
	snapshot_page integer,
    created timestamp without time zone default (now() at time zone 'utc')
)
;

create table uploads
(
	id serial not null
		constraint uploads_pkey
			primary key,
	bucket text,
	key text,
	uploadid text,
    created timestamp without time zone default (now() at time zone 'utc')
)
;

create table datasets
(
	dataset_id serial not null
		constraint datasets_pkey
			primary key,
	dataset_name text,
	view_name text,
    created timestamp without time zone default (now() at time zone 'utc')
)
;

create table systems
(
	system_uuid uuid not null
		constraint systems_pkey
			primary key,
	vendor text,
	dnshostname text,
	username text,
	systemtype text,
	totalmem bigint,
    created timestamp without time zone default (now() at time zone 'utc')
)
;

create table system_graphics
(
	system_graphic_id serial not null
		constraint system_graphics_pkey
			primary key,
	deviceid text,
	adaptercompatibility text,
	adapterdactype text,
	adapterram integer,
	availability integer,
	caption text,
	description text,
	driverdate timestamp with time zone,
	driverversion text,
	pnpdeviceid text,
	name text,
	videoarch integer,
	memtype integer,
	videoprocessor text,
	bpp integer,
	hrez integer,
	vrez integer,
	num_colors integer,
	cols integer,
	rows integer,
	refresh integer,
	scanmode integer,
	videomodedesc text,
    created timestamp without time zone default (now() at time zone 'utc')
)
;