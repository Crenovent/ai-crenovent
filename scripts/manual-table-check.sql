-- =====================================================
-- Manual Table Check - Simple Commands
-- Run these one by one to find your users table
-- =====================================================

-- Command 1: Show current database and user
SELECT current_database(), current_user;

-- Command 2: List all tables in public schema using pg_tables
SELECT tablename FROM pg_tables WHERE schemaname = 'public';

-- Command 3: List all tables using information_schema
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';

-- Command 4: Search for any table with 'user' in the name
SELECT table_schema, table_name 
FROM information_schema.tables 
WHERE table_name ILIKE '%user%';

-- Command 5: Try to access the table directly (replace 'users' with your actual table name)
SELECT COUNT(*) FROM users;

-- Command 6: If above fails, try with quotes (case sensitive)
SELECT COUNT(*) FROM "users";

-- Command 7: Try uppercase
SELECT COUNT(*) FROM "USERS";

-- Command 8: Show table structure if found (replace 'users' with actual name)
\d users

-- Command 9: Alternative way to show structure
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'users' AND table_schema = 'public';

-- Command 10: Show all schemas
SELECT schema_name FROM information_schema.schemata;