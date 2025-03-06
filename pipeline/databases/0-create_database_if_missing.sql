-- Script to create database if not exists

--creation of the database
CREATE DATABASE IF NOT EXISTS db_0;
DELIMETER //
CREATE PROCEDURE create_db_0_if_not_exists()
BEGIN
	DECLARE continue_handler INT DEFAULT 0;
	CREATE DATABASE db_0;
	DECLARE CONTINUE HANDLER FOR SQLSTATE '42000'
	     SET continue_handler = 1;
	CREATE DATABASE db_0;
END //
CALL create_db_0_if_not_exists();
