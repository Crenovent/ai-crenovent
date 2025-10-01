#!/usr/bin/env python3
"""
Hierarchy-Aware Intelligence Service
===================================

Implements hierarchy-aware data access and intelligence calculations that match
your backend's exact patterns from Pipeline and Forecast modules.

Matches patterns from:
- crenovent-backend/controller/pipeline/index.js getAccessibleUserIds
- crenovent-backend/controller/Forecast/index.js hierarchy access logic
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncpg

logger = logging.getLogger(__name__)

class HierarchyIntelligenceService:
    """
    Hierarchy-aware intelligence service matching your backend patterns exactly
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Role-based access levels (matches your backend exactly)
        self.role_access_map = {
            'admin': 'all',      # Access to all users
            'cro': 'all',        # Access to all users  
            'revenue_manager': 'all',  # Access to all users
            'svp': 'regional',   # Access to regional subordinates
            'vp': 'area',        # Access to area subordinates
            'director': 'team',  # Access to direct team
            'manager': 'team',   # Access to direct team
            'sales': 'individual' # Access to own data only
        }
    
    async def get_accessible_user_ids(self, current_user: Dict[str, Any], level_id: Optional[str] = None) -> List[int]:
        """
        Get accessible user IDs based on current user's hierarchy position and RBAC
        EXACT match to crenovent-backend/controller/pipeline/index.js getAccessibleUserIds
        """
        try:
            user_role = current_user.get('role_name') or current_user.get('role', 'sales')
            tenant_id = current_user.get('tenant_id')
            user_id = current_user.get('user_id')
            
            access_level = self.role_access_map.get(user_role, 'individual')
            self.logger.info(f"🔐 User {user_id} ({user_role}) has {access_level} access level")
            
            async with self.pool_manager.get_connection().acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                # If individual access only, return just current user
                if access_level == 'individual':
                    return [user_id]
                
                # If full access, get all active users in tenant
                if access_level == 'all':
                    all_users_query = """
                        SELECT user_id 
                        FROM users 
                        WHERE tenant_id = $1
                          AND is_active = true
                    """
                    result = await conn.fetch(all_users_query, tenant_id)
                    user_ids = [row['user_id'] for row in result]
                    self.logger.info(f"🔐 Admin/CRO access: returning {len(user_ids)} users")
                    return user_ids
                
                # For hierarchical access, build subordinate tree
                subordinate_user_ids = await self._get_subordinate_user_ids(conn, current_user, access_level)
                
                # Apply drill-down filtering if specified
                if level_id and level_id.strip():
                    filtered_user_ids = await self._apply_level_filtering(conn, subordinate_user_ids, level_id)
                    self.logger.info(f"🎯 Level filtering applied: {len(subordinate_user_ids)} -> {len(filtered_user_ids)} users")
                    return filtered_user_ids
                
                return subordinate_user_ids
                
        except Exception as e:
            self.logger.error(f"❌ Error getting accessible user IDs: {e}")
            return [current_user.get('user_id', 0)]
    
    async def _get_subordinate_user_ids(self, conn: asyncpg.Connection, current_user: Dict[str, Any], access_level: str) -> List[int]:
        """
        Get subordinate user IDs based on hierarchy
        Matches your backend's subordinate logic
        """
        try:
            user_id = current_user.get('user_id')
            
            # Get users who report to this user (direct reports)
            # Handle potential column name variations
            direct_reports_query = """
                SELECT user_id, username, email
                FROM users 
                WHERE reports_to = $1 
                  AND (is_active = true OR is_activated = true)
            """
            
            try:
                direct_reports = await conn.fetch(direct_reports_query, user_id)
                user_ids = [user_id]  # Include self
            except Exception as e:
                self.logger.error(f"❌ Error checking hierarchy access: {e}")
                # Fallback to individual access only
                return [user_id]
            
            for report in direct_reports:
                user_ids.append(report['user_id'])
                
                # For team/area/regional access, recursively get their reports too
                if access_level in ['regional', 'area', 'team']:
                    indirect_reports = await self._get_indirect_reports(conn, report['user_id'])
                    user_ids.extend(indirect_reports)
            
            self.logger.info(f"🏗️ Hierarchical access: {len(user_ids)} users (including self)")
            return list(set(user_ids))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"❌ Error getting subordinate users: {e}")
            return [current_user.get('user_id', 0)]
    
    async def _get_indirect_reports(self, conn: asyncpg.Connection, manager_id: int) -> List[int]:
        """Get indirect reports recursively"""
        try:
            indirect_query = """
                SELECT user_id 
                FROM users 
                WHERE reports_to = $1 
                  AND is_active = true
            """
            
            result = await conn.fetch(indirect_query, manager_id)
            indirect_ids = []
            
            for row in result:
                indirect_ids.append(row['user_id'])
                # Recursively get their reports too (up to reasonable depth)
                sub_reports = await self._get_indirect_reports(conn, row['user_id'])
                indirect_ids.extend(sub_reports)
            
            return indirect_ids
            
        except Exception as e:
            self.logger.error(f"❌ Error getting indirect reports: {e}")
            return []
    
    async def _apply_level_filtering(self, conn: asyncpg.Connection, user_ids: List[int], level_id: str) -> List[int]:
        """
        Apply geographic level filtering (matches your frontend level_id pattern)
        Supports level_id formats like: 'area_north-america_america', 'region_america', etc.
        """
        try:
            if not level_id or level_id == 'global':
                return user_ids
            
            # Parse level_id (e.g., 'area_north-america_america' -> level='area', value='north-america')
            parts = level_id.split('_')
            if len(parts) < 2:
                return user_ids
            
            level_type = parts[0]  # 'area', 'region', 'segment', etc.
            level_value = parts[1] if len(parts) > 1 else None
            
            # Build WHERE clause based on level type
            where_clause = ""
            if level_type == 'region':
                where_clause = "ur.region = $2"
            elif level_type == 'area':
                where_clause = "ur.area = $2"
            elif level_type == 'segment':
                where_clause = "ur.segment = $2"
            elif level_type == 'district':
                where_clause = "ur.district = $2"
            elif level_type == 'territory':
                where_clause = "ur.territory = $2"
            else:
                return user_ids  # Unknown level type, return all
            
            # Query users with geographic filtering
            filtered_query = f"""
                SELECT DISTINCT u.user_id
                FROM users u
                LEFT JOIN users_role ur ON u.user_id = ur.user_id
                WHERE u.user_id = ANY($1)
                  AND {where_clause}
            """
            
            result = await conn.fetch(filtered_query, user_ids, level_value)
            filtered_ids = [row['user_id'] for row in result]
            
            self.logger.info(f"🎯 Level filtering ({level_type}={level_value}): {len(user_ids)} -> {len(filtered_ids)} users")
            return filtered_ids
            
        except Exception as e:
            self.logger.error(f"❌ Error applying level filtering: {e}")
            return user_ids
    
    async def check_hierarchy_access(self, current_user: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if user has hierarchy access (matches your frontend pattern)
        Returns (has_access, access_reason)
        """
        try:
            user_role = current_user.get('role_name', '')
            user_profile = current_user.get('profile', {})
            user_id = current_user.get('user_id')
            tenant_id = current_user.get('tenant_id')
            
            # Check if user is admin or revenue_manager
            if user_role in ['admin', 'revenue_manager']:
                return True, f"role_{user_role}"
            
            # Check if user level is m1-m7
            user_level = user_profile.get('level', '')
            if user_level in ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']:
                return True, f"level_{user_level}"
            
            # Check if user has reporting users
            async with self.pool_manager.get_connection().acquire() as conn:
                await conn.execute(f"SET app.current_tenant = '{tenant_id}'")
                
                reports_query = """
                    SELECT COUNT(*) as report_count
                    FROM users 
                    WHERE reports_to = $1 
                      AND tenant_id = $2
                      AND is_active = true
                """
                
                result = await conn.fetchrow(reports_query, user_id, tenant_id)
                has_reporting_users = int(result['report_count'] or 0) > 0
                
                if has_reporting_users:
                    return True, "has_reports"
            
            return False, "individual_only"
            
        except Exception as e:
            self.logger.error(f"❌ Error checking hierarchy access: {e}")
            return False, "error"
    
    async def get_user_hierarchy_context(self, current_user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user's hierarchy context for intelligence calculations
        """
        try:
            has_hierarchy_access, access_reason = await self.check_hierarchy_access(current_user)
            accessible_user_ids = await self.get_accessible_user_ids(current_user)
            
            return {
                'user_id': current_user.get('user_id'),
                'tenant_id': current_user.get('tenant_id'),
                'role_name': current_user.get('role_name'),
                'has_hierarchy_access': has_hierarchy_access,
                'access_reason': access_reason,
                'accessible_user_ids': accessible_user_ids,
                'access_scope': 'team' if len(accessible_user_ids) > 1 else 'individual',
                'total_accessible_users': len(accessible_user_ids)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting hierarchy context: {e}")
            return {
                'user_id': current_user.get('user_id'),
                'tenant_id': current_user.get('tenant_id'),
                'has_hierarchy_access': False,
                'access_reason': 'error',
                'accessible_user_ids': [current_user.get('user_id')],
                'access_scope': 'individual',
                'total_accessible_users': 1
            }
    
    async def log_hierarchy_access(self, current_user: Dict[str, Any], level_id: Optional[str], result_count: int):
        """
        Log hierarchy access for audit and knowledge graph
        """
        try:
            async with self.pool_manager.get_connection().acquire() as conn:
                # Log to dsl_execution_traces for knowledge graph
                trace_query = """
                INSERT INTO dsl_execution_traces (
                    workflow_id, tenant_id, user_id, execution_type,
                    input_data, output_data, execution_status,
                    started_at, completed_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """
                
                workflow_id = f"hierarchy_access_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                
                await conn.execute(
                    trace_query,
                    workflow_id,
                    current_user.get('tenant_id'),
                    current_user.get('user_id'),
                    'hierarchy_access',
                    {
                        'level_id': level_id,
                        'user_role': current_user.get('role_name'),
                        'request_time': datetime.utcnow().isoformat()
                    },
                    {'result_count': result_count},
                    'completed',
                    datetime.utcnow(),
                    datetime.utcnow(),
                    {
                        'source': 'hierarchy_intelligence_service',
                        'access_pattern': 'hierarchy_aware_intelligence'
                    }
                )
                
                self.logger.info(f"📊 Logged hierarchy access to knowledge graph: {workflow_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Error logging hierarchy access: {e}")
