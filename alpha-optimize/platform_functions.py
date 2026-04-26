#!/usr/bin/env python3
"""
WorldQuant BRAIN MCP Server - Python Version
A comprehensive Model Context Protocol (MCP) server for WorldQuant BRAIN platform integration.
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import re
import base64
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import sys
import math
import io
# try set gbk problem
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    elif hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

import requests
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, EmailStr

from pathlib import Path

# Import the new forum client
from forum_functions import forum_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for type safety
class AuthCredentials(BaseModel):
    email: EmailStr
    password: str

class SimulationSettings(BaseModel):
    instrumentType: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: float = 0.0
    neutralization: str = "NONE"
    truncation: float = 0.0
    pasteurization: str = "ON"
    unitHandling: str = "VERIFY"
    nanHandling: str = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = True
    testPeriod: str = "P0Y0M"
    selectionHandling: str = "POSITIVE"
    selectionLimit: int = 1000
    maxTrade: str = "OFF"
    maxPosition: str = "OFF"
    componentActivation: str = "IS"

class SimulationData(BaseModel):
    type: str = "REGULAR"  # "REGULAR" or "SUPER"
    settings: SimulationSettings
    regular: Optional[str] = None
    combo: Optional[str] = None
    selection: Optional[str] = None

class BrainApiClient:
    """WorldQuant BRAIN API client with comprehensive functionality."""
    
    def __init__(self):
        self.base_url = "https://api.worldquantbrain.com"
        self.session = requests.Session()
        self.auth_credentials = None
        self.is_authenticating = False
        
        self._auth_lock = asyncio.Lock()

        # Configure session
        self.session.timeout = 30
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def log(self, message: str, level: str = "INFO"):
        """Log messages to stderr to avoid MCP protocol interference."""
        try:
            # Try to print with original message first
            print(f"[{level}] {message}", file=sys.stderr)
        except UnicodeEncodeError:
            # Fallback: remove problematic characters and try again
            try:
                safe_message = message.encode('ascii', 'ignore').decode('ascii')
                print(f"[{level}] {safe_message}", file=sys.stderr)
            except Exception:
                # Final fallback: just print the level and a safe message
                print(f"[{level}] Log message", file=sys.stderr)
        except Exception:
            # Final fallback: just print the level and a safe message
            print(f"[{level}] Log message", file=sys.stderr)

    async def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Run a synchronous requests call in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        func = getattr(self.session, method)
        return await loop.run_in_executor(None, lambda: func(url, **kwargs))

    async def authenticate(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate with WorldQuant BRAIN platform with biometric support."""
        self.log("🔐 Starting Authentication process...", "INFO")
        
        try:
            # Store credentials for potential re-authentication
            self.auth_credentials = {'email': email, 'password': password}
            
            # Clear any existing session data
            self.session.cookies.clear()
            self.session.auth = None
            
            # Create Basic Authentication header (base64 encoded credentials)
            import base64
            credentials = f"{email}:{password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            # Send POST request with Basic Authentication header
            headers = {
                'Authorization': f'Basic {encoded_credentials}'
            }
            
            response = await self._request('post', 'https://api.worldquantbrain.com/authentication', headers=headers)
            
            # Check for successful authentication (status code 201)
            if response.status_code == 201:
                self.log("Authentication successful", "SUCCESS")
                
                # Check if JWT token was automatically stored by session
                jwt_token = self.session.cookies.get('t')
                if jwt_token:
                    self.log("JWT token automatically stored by session", "SUCCESS")
                else:
                    self.log("⚠️ No JWT token found in session", "WARNING")
                
                # Return success response
                return {
                    'user': {'email': email},
                    'status': 'authenticated',
                    'permissions': ['read', 'write'],
                    'message': 'Authentication successful',
                    'status_code': response.status_code,
                    'has_jwt': jwt_token is not None
                }
            
            # Check if biometric authentication is required (401 with persona)
            elif response.status_code == 401:
                www_auth = response.headers.get("WWW-Authenticate")
                location = response.headers.get("Location")
                
                if www_auth == "persona" and location:
                    self.log("🔴 Biometric authentication required", "INFO")
                    
                    # Handle biometric authentication
                    from urllib.parse import urljoin
                    biometric_url = urljoin(response.url, location)
                    
                    return await self._handle_biometric_auth(biometric_url, email)
                else:
                    raise Exception("Incorrect email or password")
            else:
                raise Exception(f"Authentication failed with status code: {response.status_code}")
                    
        except requests.HTTPError as e:
            self.log(f"❌ HTTP error during authentication: {e}", "ERROR")
            raise
        except Exception as e:
            self.log(f"❌ Authentication failed: {str(e)}", "ERROR")
            raise
    
    async def _handle_biometric_auth(self, biometric_url: str, email: str) -> Dict[str, Any]:
        """Handle biometric authentication using browser automation."""
        self.log("🌐 Starting biometric authentication...", "INFO")
        
        try:
            # Import playwright for browser automation
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False)
                page = await browser.new_page()

                self.log("🌐 Opening browser for biometric authentication...", "INFO")
                await page.goto(biometric_url)
                self.log("Browser page loaded successfully", "SUCCESS")

                # Print instructions
                print("\n" + "="*60, file=sys.stderr)
                print("BIOMETRIC AUTHENTICATION REQUIRED", file=sys.stderr)
                print("="*60, file=sys.stderr)
                print("Browser window is open with biometric authentication page", file=sys.stderr)
                print("Complete the biometric authentication in the browser", file=sys.stderr)
                print("The system will automatically check when you're done...", file=sys.stderr)
                print("="*60, file=sys.stderr)

                # Keep checking until authentication is complete
                max_attempts = 60  # 5 minutes maximum (60 * 5 seconds)
                attempt = 0

                while attempt < max_attempts:
                    await asyncio.sleep(5)  # Check every 5 seconds
                    attempt += 1

                    # Check if authentication completed
                    check_response = await self._request('post', biometric_url)
                    self.log(f"🔄 Checking authentication status (attempt {attempt}/{max_attempts}): {check_response.status_code}", "INFO")

                    if check_response.status_code == 201:
                        self.log("Biometric authentication successful!", "SUCCESS")

                        await browser.close()
                        
                        # Check JWT token
                        jwt_token = self.session.cookies.get('t')
                        if jwt_token:
                            self.log("JWT token received", "SUCCESS")
                        
                        # Return success response
                        return {
                            'user': {'email': email},
                            'status': 'authenticated',
                            'permissions': ['read', 'write'],
                            'message': 'Biometric authentication successful',
                            'status_code': check_response.status_code,
                            'has_jwt': jwt_token is not None
                        }
                
                await browser.close()
                raise Exception("Biometric authentication timed out")

        except Exception as e:
            self.log(f"❌ Biometric authentication failed: {str(e)}", "ERROR")
            raise
    
    async def is_authenticated(self) -> bool:
        """Check if currently authenticated using JWT token."""
        try:
            # Check if we have a JWT token in cookies
            jwt_token = self.session.cookies.get('t')
            if not jwt_token:
                self.log("❌ No JWT token found", "INFO")
                return False
            
            # Test authentication with a simple API call
            response = await self._request('get', f"{self.base_url}/authentication")
            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                self.log("❌ JWT token expired or invalid (401)", "INFO")
                return False
            else:
                self.log(f"⚠️ Unexpected status code during auth check: {response.status_code}", "WARNING")
                return False
        except Exception as e:
            self.log(f"❌ Error checking authentication: {str(e)}", "ERROR")
            return False
    
    async def ensure_authenticated(self):
        """Ensure authentication is valid, re-authenticate if needed."""
        async with self._auth_lock:
            if not await self.is_authenticated():
                if not self.auth_credentials:
                    self.log("No credentials in memory, loading from config...", "INFO")
                    config = load_config()
                    creds = config.get("credentials", {})
                    email = creds.get("email")
                    password = creds.get("password")
                    if not email or not password:
                        raise Exception("Authentication credentials not found in config. Please authenticate first.")
                    self.auth_credentials = {'email': email, 'password': password}

                self.log("🔄 Re-authenticating...", "INFO")
                await self.authenticate(self.auth_credentials['email'], self.auth_credentials['password'])
    
    async def get_authentication_status(self) -> Optional[Dict[str, Any]]:
        """Get current authentication status and user info."""
        try:
            response = await self._request('get', f"{self.base_url}/users/self")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get auth status: {str(e)}", "ERROR")
            return None
    
    async def create_simulation(self, simulation_data: SimulationData) -> Dict[str, str]:
        """Create a new simulation on BRAIN platform."""
        await self.ensure_authenticated()
        
        try:
            self.log("🚀 Creating simulation...", "INFO")
            
            # Prepare settings based on simulation type
            settings_dict = simulation_data.settings.model_dump()
            
            # Remove fields based on simulation type
            if simulation_data.type == "REGULAR":
                # Remove SUPER-specific fields for REGULAR
                settings_dict.pop('selectionHandling', None)
                settings_dict.pop('selectionLimit', None)
                settings_dict.pop('componentActivation', None)
            
            # Filter out None values from settings
            settings_dict = {k: v for k, v in settings_dict.items() if v is not None}
            
            # Prepare simulation payload
            payload = {
                'type': simulation_data.type,
                'settings': settings_dict
            }
            
            # Add type-specific fields
            if simulation_data.type == "REGULAR":
                if simulation_data.regular:
                    payload['regular'] = simulation_data.regular
            elif simulation_data.type == "SUPER":
                if simulation_data.combo:
                    payload['combo'] = simulation_data.combo
                if simulation_data.selection:
                    payload['selection'] = simulation_data.selection
            
            # Filter out None values from entire payload
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = await self._request('post', f"{self.base_url}/simulations", json=payload)
            response.raise_for_status()

            location = response.headers.get('Location', '')
            simulation_id = location.split('/')[-1] if location else None

            self.log(f"Simulation created with ID: {simulation_id}", "SUCCESS")

            while True:
                simulation_progress = await self._request('get', location)
                if simulation_progress.headers.get("Retry-After", 0) == 0:
                    break
                print("Sleeping for " + simulation_progress.headers["Retry-After"] + " seconds")
                await asyncio.sleep(float(simulation_progress.headers["Retry-After"]))
            print("Alpha done simulating, getting alpha details")
            alpha_id = simulation_progress.json()["alpha"]
            alpha = await self._request('get', "https://api.worldquantbrain.com/alphas/" + alpha_id)
            result = alpha.json()
            result['note'] = "if you got a negative alpha sharpe, you can just add a minus sign in front of the last line of the Alpha to flip then think the next step."
            return result
            
        except Exception as e:
            self.log(f"❌ Failed to create simulation: {str(e)}", "ERROR")
            raise
    
    async def get_alpha_details(self, alpha_id: str) -> Dict[str, Any]:
        """Get detailed information about an alpha."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get alpha details: {str(e)}", "ERROR")
            raise
    
    async def get_datasets(self, instrument_type: str = "EQUITY", region: str = "USA",
                          delay: int = 1, universe: str = "TOP3000", theme: str = "false", search: Optional[str] = None) -> Dict[str, Any]:
        """Get available datasets."""
        await self.ensure_authenticated()
        
        try:
            params = {
                'instrumentType': instrument_type,
                'region': region,
                'delay': delay,
                'universe': universe,
                'theme': theme
            }
            
            if search:
                params['search'] = search
            
            response = await self._request('get', f"{self.base_url}/data-sets", params=params)
            response.raise_for_status()
            response_json = response.json()
            response_json['extraNote'] = "if your returned result is 0, you may want to check your parameter by using get_platform_setting_options tool to got correct parameter"
            return response_json
        except Exception as e:
            self.log(f"Failed to get datasets: {str(e)}", "ERROR")
            raise
    
    async def get_datafields(self, instrument_type: str = "EQUITY", region: str = "USA",
                            delay: int = 1, universe: str = "TOP3000", theme: str = "false",
                            dataset_id: Optional[str] = None, data_type: str = "",
                            search: Optional[str] = None) -> Dict[str, Any]:
        """Get available data fields."""
        await self.ensure_authenticated()
        
        try:
            params = {
                'instrumentType': instrument_type,
                'region': region,
                'delay': delay,
                'universe': universe,
                'limit': '50',
                'offset': '0'
            }
            
            if data_type != 'ALL':
                params['type'] = data_type
            
            if dataset_id:
                params['dataset.id'] = dataset_id
            if search:
                params['search'] = search
            
            response = await self._request('get', f"{self.base_url}/data-fields", params=params)
            response.raise_for_status()
            response_json = response.json()
            response_json['extraNote'] = "if your returned result is 0, you may want to check your parameter by using get_platform_setting_options tool to got correct parameter"
            return response_json
        except Exception as e:
            self.log(f"Failed to get datafields: {str(e)}", "ERROR")
            raise
    
    async def get_alpha_pnl(self, alpha_id: str) -> Dict[str, Any]:
        """Get PnL data for an alpha with retry logic."""
        await self.ensure_authenticated()
        
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.log(f"Attempting to get PnL for alpha {alpha_id} (attempt {attempt + 1}/{max_retries})", "INFO")
                
                response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/recordsets/pnl")
                response.raise_for_status()

                # Some alphas may return 204 No Content or an empty body
                text = (response.text or "").strip()
                if not text:
                    if attempt < max_retries - 1:
                        self.log(f"Empty PnL response for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                        continue
                    else:
                        self.log(f"Empty PnL response after {max_retries} attempts for {alpha_id}", "WARNING")
                        return {}
                
                try:
                    pnl_data = response.json()
                    if pnl_data:
                        self.log(f"Successfully retrieved PnL data for alpha {alpha_id}", "SUCCESS")
                        return pnl_data
                    else:
                        if attempt < max_retries - 1:
                            self.log(f"Empty PnL JSON for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 1.5
                            continue
                        else:
                            self.log(f"Empty PnL JSON after {max_retries} attempts for {alpha_id}", "WARNING")
                            return {}
                            
                except Exception as parse_err:
                    if attempt < max_retries - 1:
                        self.log(f"PnL JSON parse failed for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    else:
                        self.log(f"PnL JSON parse failed for {alpha_id} after {max_retries} attempts: {parse_err}", "WARNING")
                        return {}
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    self.log(f"Failed to get alpha PnL for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds: {str(e)}", "WARNING")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    self.log(f"Failed to get alpha PnL for {alpha_id} after {max_retries} attempts: {str(e)}", "ERROR")
                    raise
        
        # This should never be reached, but just in case
        return {}
    
    async def get_user_alphas(
        self,
        stage: str = "OS",
        limit: int = 30,
        offset: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        submission_start_date: Optional[str] = None,
        submission_end_date: Optional[str] = None,
        order: Optional[str] = None,
        hidden: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Get user's alphas with advanced filtering."""
        await self.ensure_authenticated()
        
        try:
            params = {
                "stage": stage,
                "limit": limit,
                "offset": offset,
            }
            if start_date:
                params["dateCreated>"] = start_date
            if end_date:
                params["dateCreated<"] = end_date
            if submission_start_date:
                params["dateSubmitted>"] = submission_start_date
            if submission_end_date:
                params["dateSubmitted<"] = submission_end_date
            if order:
                params["order"] = order
            if hidden is not None:
                params["hidden"] = str(hidden).lower()

            response = await self._request('get', f"{self.base_url}/users/self/alphas", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get user alphas: {str(e)}", "ERROR")
            raise
    
    async def submit_alpha(self, alpha_id: str) -> bool:
        """Submit an alpha for production."""
        await self.ensure_authenticated()
        
        try:
            self.log(f"📤 Submitting alpha {alpha_id} for production...", "INFO")
            
            response = await self._request('post', f"{self.base_url}/alphas/{alpha_id}/submit")
            response.raise_for_status()

            self.log(f"Alpha {alpha_id} submitted successfully", "SUCCESS")
            return response.__dict__
            
        except Exception as e:
            self.log(f"❌ Failed to submit alpha: {str(e)}", "ERROR")
            return False
    
    async def get_events(self) -> Dict[str, Any]:
        """Get available events and competitions."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/events")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get events: {str(e)}", "ERROR")
            raise
    
    async def get_leaderboard(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get leaderboard data."""
        await self.ensure_authenticated()
        
        try:
            params = {}
            
            if user_id:
                params['user'] = user_id
            else:
                # Get current user ID if not specified
                user_response = await self._request('get', f"{self.base_url}/users/self")
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    params['user'] = user_data.get('id')

            response = await self._request('get', f"{self.base_url}/consultant/boards/leader", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get leaderboard: {str(e)}", "ERROR")
            raise

    def _is_atom(self, detail: Optional[Dict[str, Any]]) -> bool:
        """Match atom detection used in extract_regular_alphas.py:
        - Primary signal: 'classifications' entries containing 'SINGLE_DATA_SET'
        - Fallbacks: tags list contains 'atom' or classification id/name contains 'ATOM'
        """
        if not detail or not isinstance(detail, dict):
            return False

        classifications = detail.get('classifications') or []
        for c in classifications:
            cid = (c.get('id') or c.get('name') or '')
            if isinstance(cid, str) and 'SINGLE_DATA_SET' in cid:
                return True

        # Fallbacks
        tags = detail.get('tags') or []
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, str) and t.strip().lower() == 'atom':
                    return True

        for c in classifications:
            cid = (c.get('id') or c.get('name') or '')
            if isinstance(cid, str) and 'ATOM' in cid.upper():
                return True

        return False

    async def value_factor_trendScore(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Compute diversity score for regular alphas in a date range.

        Description:
        This function calculate the diversity of the users' submission, by checking the diversity, we can have a good understanding on the valuefactor's trend.
        value factor of a user is defiend by This diversity score, which measures three key aspects of work output: the proportion of works
        with the "Atom" tag (S_A), atom proportion, the breadth of pyramids covered (S_P), and how evenly works
        are distributed across those pyramids (S_H). Calculated as their product, it rewards
        strong performance across all three dimensions—encouraging more Atom-tagged works,
        wider pyramid coverage, and balanced distribution—with weaknesses in any area lowering
        the total score significantly.

        Inputs (hints for AI callers):
        - start_date (str): ISO UTC start datetime, e.g. '2025-08-14T00:00:00Z'
        - end_date (str): ISO UTC end datetime, e.g. '2025-08-18T23:59:59Z'
        - Note: this tool always uses 'OS' (submission dates) to define the window; callers do not need to supply a stage.
                - Note: P_max (total number of possible pyramids) is derived from the platform
                    pyramid-multipliers endpoint and not supplied by callers.

        Returns (compact JSON): {
            'diversity_score': float,
            'N': int,  # total regular alphas in window
            'A': int,  # number of Atom-tagged works (is_single_data_set)
            'P': int,  # pyramid coverage count in the sample
            'P_max': int, # used max for normalization
            'S_A': float, 'S_P': float, 'S_H': float,
            'per_pyramid_counts': {pyramid_name: count}
        }
        """
        # Fetch user alphas (always use OS / submission dates per product policy)
        await self.ensure_authenticated()
        alphas_resp = await self.get_user_alphas(stage='OS', limit=500, submission_start_date=start_date, submission_end_date=end_date)

        if not isinstance(alphas_resp, dict) or 'results' not in alphas_resp:
            return {'error': 'Unexpected response from get_user_alphas', 'raw': alphas_resp}

        alphas = alphas_resp['results']
        regular = [a for a in alphas if a.get('type') == 'REGULAR']

        # Fetch details for each regular alpha
        pyramid_list = []
        atom_count = 0
        per_pyramid = {}
        for a in regular:
            try:
                detail = await self.get_alpha_details(a.get('id'))
            except Exception:
                continue

            is_atom = self._is_atom(detail)
            if is_atom:
                atom_count += 1

            # Extract pyramids
            ps = []
            if isinstance(detail.get('pyramids'), list):
                ps = [p.get('name') for p in detail.get('pyramids') if p.get('name')]
            else:
                pt = detail.get('pyramidThemes') or {}
                pss = pt.get('pyramids') if isinstance(pt, dict) else None
                if pss and isinstance(pss, list):
                    ps = [p.get('name') for p in pss if p.get('name')]

            for p in ps:
                pyramid_list.append(p)
                per_pyramid[p] = per_pyramid.get(p, 0) + 1

        N = len(regular)
        A = atom_count
        P = len(per_pyramid)

        # Determine P_max similarly to the script: use pyramid multipliers if available
        P_max = None
        try:
            pm = await self.get_pyramid_multipliers()
            if isinstance(pm, dict) and 'pyramids' in pm:
                pyramids_list = pm.get('pyramids') or []
                P_max = len(pyramids_list)
        except Exception:
            P_max = None

        if not P_max or P_max <= 0:
            P_max = max(P, 1)

        # Component scores
        S_A = (A / N) if N > 0 else 0.0
        S_P = (P / P_max) if P_max > 0 else 0.0

        # Entropy
        S_H = 0.0
        if P <= 1 or not per_pyramid:
            S_H = 0.0
        else:
            total_occ = sum(per_pyramid.values())
            H = 0.0
            for cnt in per_pyramid.values():
                q = cnt / total_occ if total_occ > 0 else 0
                if q > 0:
                    H -= q * math.log2(q)
            max_H = math.log2(P) if P > 0 else 1
            S_H = (H / max_H) if max_H > 0 else 0.0

        diversity_score = S_A * S_P * S_H

        return {
            'diversity_score': diversity_score,
            'N': N,
            'A': A,
            'P': P,
            'P_max': P_max,
            'S_A': S_A,
            'S_P': S_P,
            'S_H': S_H,
            'per_pyramid_counts': per_pyramid
        }

    async def get_operators(self) -> Dict[str, Any]:
        """Get available operators for alpha creation."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/operators")
            response.raise_for_status()
            operators_data = response.json()
            
            # Ensure we return a dictionary format even if API returns a list
            if isinstance(operators_data, list):
                return {"operators": operators_data, "count": len(operators_data)}
            else:
                return operators_data
        except Exception as e:
            self.log(f"Failed to get operators: {str(e)}", "ERROR")
            raise
            
    async def run_selection(
        self,
        selection: str,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        delay: int = 1,
        selection_limit: int = 1000,
        selection_handling: str = "POSITIVE"
    ) -> Dict[str, Any]:
        """Run a selection query to filter instruments."""
        await self.ensure_authenticated()
        
        try:
            selection_data = {
                "selection": selection,
                "instrumentType": instrument_type,
                "region": region,
                "delay": delay,
                "selectionLimit": selection_limit,
                "selectionHandling": selection_handling
            }
            
            response = await self._request('get', f"{self.base_url}/simulations/super-selection", params=selection_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to run selection: {str(e)}", "ERROR")
            raise

    async def get_user_profile(self, user_id: str = "self") -> Dict[str, Any]:
        """Get user profile information."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/users/{user_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get user profile: {str(e)}", "ERROR")
            raise
            
    async def get_documentations(self) -> Dict[str, Any]:
        """Get available documentations and learning materials."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/tutorials")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get documentations: {str(e)}", "ERROR")
            raise
            
    async def get_messages(self, limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
        """Get messages for the current user with optional pagination.

        Image / large binary payload mitigation:
          Some messages embed base64 encoded images (e.g. <img src="data:image/png;base64,..."/>).
          Returning full base64 can explode token usage for an LLM client. We post-process each
          message description and (by default) extract embedded base64 images to disk and replace
          them with lightweight placeholders while preserving context.

        Strategies (environment driven in future – currently parameterless public API):
          - placeholder (default): save images to message_images/ and replace with marker text.
          - ignore: strip image tags entirely, leaving a note.
          - keep: leave description unchanged (unsafe for LLM token limits).

        A message dict gains an 'extracted_images' list when images are processed.
        """
        await self.ensure_authenticated()

        import re, base64, pathlib

        image_handling = os.environ.get("BRAIN_MESSAGE_IMAGE_MODE", "placeholder").lower()
        save_dir = pathlib.Path("message_images")

        from typing import Tuple
        def process_description(desc: str, message_id: str) -> Tuple[str, List[str]]:
            try:
                if not desc or image_handling == "keep":
                    return desc, []
                attachments: List[str] = []
                # Regex to capture full <img ...> tag with data URI
                img_tag_pattern = re.compile(r"<img[^>]+src=\"(data:image/[^\"]+)\"[^>]*>", re.IGNORECASE)
                # Iterate over unique matches to avoid double work
                matches = list(img_tag_pattern.finditer(desc))
                if not matches:
                    # Additional heuristic: very long base64-looking token inside quotes followed by </img>
                    # (legacy format noted by user sample). Replace with placeholder.
                    heuristic_pattern = re.compile(r"([A-Za-z0-9+/]{500,}={0,2})\"\s*</img>")
                    if image_handling != "keep" and heuristic_pattern.search(desc):
                        placeholder = "[Embedded image removed - large base64 sequence truncated]"
                        return heuristic_pattern.sub(placeholder + "</img>", desc), []
                    return desc, []

                # Ensure save directory exists only if we will store something
                if image_handling == "placeholder" and not save_dir.exists():
                    try:
                        save_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        self.log(f"Could not create image save directory: {e}", "WARNING")

                new_desc = desc
                for idx, match in enumerate(matches, start=1):
                    data_uri = match.group(1)  # data:image/...;base64,XXXX
                    if not data_uri.lower().startswith("data:image"):
                        continue
                    # Split header and base64 payload
                    if "," not in data_uri:
                        continue
                    header, b64_data = data_uri.split(",", 1)
                    mime_part = header.split(";")[0]  # data:image/png
                    ext = "png"
                    if "/" in mime_part:
                        ext = mime_part.split("/")[1]
                    safe_ext = (ext or "img").split("?")[0]
                    placeholder_text = "[Embedded image]"
                    if image_handling == "ignore":
                        replacement = f"[Image removed: {safe_ext}]"
                    elif image_handling == "placeholder":
                        # Try decode & save
                        file_name = f"{message_id}_{idx}.{safe_ext}"
                        file_path = save_dir / file_name
                        try:
                            # Guard extremely large strings (>5MB ~ 6.7M base64 chars) to avoid memory blow
                            if len(b64_data) > 7_000_000:
                                raise ValueError("Image too large to decode safely")
                            with open(file_path, "wb") as f:
                                f.write(base64.b64decode(b64_data))
                            attachments.append(str(file_path))
                            replacement = f"[Image extracted -> {file_path}]"
                        except Exception as e:
                            self.log(f"Failed to decode embedded image in message {message_id}: {e}", "WARNING")
                            replacement = "[Image extraction failed - content omitted]"
                    else:  # keep
                        replacement = placeholder_text  # shouldn't be used since early return, but safe
                    # Replace only the matched tag (not global) – use re.sub with count=1 on substring slice
                    # Safer to operate on new_desc using the exact matched string
                    original_tag = match.group(0)
                    new_desc = new_desc.replace(original_tag, replacement, 1)
                return new_desc, attachments
            except UnicodeEncodeError as ue:
                self.log(f"Unicode encoding error in process_description: {ue}", "WARNING")
                return desc, []
            except Exception as e:
                self.log(f"Error in process_description: {e}", "WARNING")
                return desc, []

        try:
            params = {}
            if limit is not None:
                params['limit'] = limit
            if offset > 0:
                params['offset'] = offset

            response = await self._request('get', f"{self.base_url}/users/self/messages", params=params)
            response.raise_for_status()
            data = response.json()

            # Post-process results for image handling
            results = data.get('results', [])
            for msg in results:
                try:
                    desc = msg.get('description')
                    processed_desc, attachments = process_description(desc, msg.get('id', 'msg'))
                    if attachments or desc != processed_desc:
                        msg['description'] = processed_desc
                        if attachments:
                            msg['extracted_images'] = attachments
                        else:
                            # If changed but no attachments (ignore mode) mark sanitized
                            msg['sanitized'] = True
                except UnicodeEncodeError as ue:
                    self.log(f"Unicode encoding error sanitizing message {msg.get('id')}: {ue}", "WARNING")
                    # Keep original description if encoding fails
                    continue
                except Exception as inner_e:
                    self.log(f"Failed to sanitize message {msg.get('id')}: {inner_e}", "WARNING")
            data['results'] = results
            data['image_handling'] = image_handling
            return data
        except UnicodeEncodeError as ue:
            self.log(f"Failed to get messages due to encoding error: {str(ue)}", "ERROR")
            raise
        except Exception as e:
            self.log(f"Failed to get messages: {str(e)}", "ERROR")
            raise

    async def get_glossary_terms(self, email: str, password: str) -> List[Dict[str, str]]:
        """Get glossary terms from forum."""
        try:
            return await forum_client.get_glossary_terms(email, password)
        except Exception as e:
            self.log(f"Failed to get glossary terms: {str(e)}", "ERROR")
            raise

    async def search_forum_posts(self, email: str, password: str, search_query: str, 
                                 max_results: int = 50) -> Dict[str, Any]:
        """Search forum posts."""
        try:
            return await forum_client.search_forum_posts(email, password, search_query, max_results)
        except Exception as e:
            self.log(f"Failed to search forum posts: {str(e)}", "ERROR")
            raise

    async def read_forum_post(self, email: str, password: str, article_id: str, 
                              include_comments: bool = True) -> Dict[str, Any]:
        """Get forum post."""
        try:
            return await forum_client.read_full_forum_post(email, password, article_id, include_comments)
        except Exception as e:
            self.log(f"Failed to read forum post: {str(e)}", "ERROR")
            raise
    
    async def get_alpha_yearly_stats(self, alpha_id: str) -> Dict[str, Any]:
        """Get yearly statistics for an alpha with retry logic."""
        await self.ensure_authenticated()
        
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.log(f"Attempting to get yearly stats for alpha {alpha_id} (attempt {attempt + 1}/{max_retries})", "INFO")
                
                response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/recordsets/yearly-stats")
                response.raise_for_status()

                # Check if response has content
                text = (response.text or "").strip()
                if not text:
                    if attempt < max_retries - 1:
                        self.log(f"Empty yearly stats response for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                        continue
                    else:
                        self.log(f"Empty yearly stats response after {max_retries} attempts for {alpha_id}", "WARNING")
                        return {}
                
                try:
                    yearly_stats = response.json()
                    if yearly_stats:
                        self.log(f"Successfully retrieved yearly stats for alpha {alpha_id}", "SUCCESS")
                        return yearly_stats
                    else:
                        if attempt < max_retries - 1:
                            self.log(f"Empty yearly stats JSON for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 1.5
                            continue
                        else:
                            self.log(f"Empty yearly stats JSON after {max_retries} attempts for {alpha_id}", "WARNING")
                            return {}
                            
                except Exception as parse_err:
                    if attempt < max_retries - 1:
                        self.log(f"Yearly stats JSON parse failed for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    else:
                        self.log(f"Yearly stats JSON parse failed for {alpha_id} after {max_retries} attempts: {parse_err}", "WARNING")
                        return {}
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    self.log(f"Failed to get alpha yearly stats for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds: {str(e)}", "WARNING")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    self.log(f"Failed to get alpha yearly stats for {alpha_id} after {max_retries} attempts: {str(e)}", "ERROR")
                    raise
        
        # This should never be reached, but just in case
        return {}
        
    async def get_production_correlation(self, alpha_id: str) -> Dict[str, Any]:
        """Get production correlation data for an alpha with retry logic."""
        await self.ensure_authenticated()
        
        max_retries = 5
        retry_delay = 20  # seconds
        
        for attempt in range(max_retries):
            try:
                self.log(f"Attempting to get production correlation for alpha {alpha_id} (attempt {attempt + 1}/{max_retries})", "INFO")
                
                response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/correlations/prod")
                response.raise_for_status()

                # Check if response has content
                text = (response.text or "").strip()
                if not text:
                    if attempt < max_retries - 1:
                        self.log(f"Empty production correlation response for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        self.log(f"Empty production correlation response after {max_retries} attempts for {alpha_id}", "WARNING")
                        return {}
                
                try:
                    correlation_data = response.json()
                    if correlation_data:
                        self.log(f"Successfully retrieved production correlation for alpha {alpha_id}", "SUCCESS")
                        return correlation_data
                    else:
                        if attempt < max_retries - 1:
                            self.log(f"Empty production correlation JSON for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            self.log(f"Empty production correlation JSON after {max_retries} attempts for {alpha_id}", "WARNING")
                            return {}
                            
                except Exception as parse_err:
                    if attempt < max_retries - 1:
                        self.log(f"Production correlation JSON parse failed for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        self.log(f"Production correlation JSON parse failed for {alpha_id} after {max_retries} attempts: {parse_err}", "WARNING")
                        return {}
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    self.log(f"Failed to get production correlation for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds: {str(e)}", "WARNING")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    self.log(f"Failed to get production correlation for {alpha_id} after {max_retries} attempts: {str(e)}", "ERROR")
                    raise
        
        # This should never be reached, but just in case
        return {}

    async def get_self_correlation(self, alpha_id: str) -> Dict[str, Any]:
        """Get self-correlation data for an alpha with retry logic."""
        await self.ensure_authenticated()
        
        max_retries = 5
        retry_delay = 20  # seconds
        
        for attempt in range(max_retries):
            try:
                self.log(f"Attempting to get self correlation for alpha {alpha_id} (attempt {attempt + 1}/{max_retries})", "INFO")
                
                response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/correlations/self")
                response.raise_for_status()

                # Check if response has content
                text = (response.text or "").strip()
                if not text:
                    if attempt < max_retries - 1:
                        self.log(f"Empty self correlation response for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        self.log(f"Empty self correlation response after {max_retries} attempts for {alpha_id}", "WARNING")
                        return {}
                
                try:
                    correlation_data = response.json()
                    if correlation_data:
                        self.log(f"Successfully retrieved self correlation for alpha {alpha_id}", "SUCCESS")
                        return correlation_data
                    else:
                        if attempt < max_retries - 1:
                            self.log(f"Empty self correlation JSON for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            self.log(f"Empty self correlation JSON after {max_retries} attempts for {alpha_id}", "WARNING")
                            return {}
                            
                except Exception as parse_err:
                    if attempt < max_retries - 1:
                        self.log(f"Self correlation JSON parse failed for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        self.log(f"Self correlation JSON parse failed for {alpha_id} after {max_retries} attempts: {parse_err}", "WARNING")
                        return {}
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    self.log(f"Failed to get self correlation for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds: {str(e)}", "WARNING")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    self.log(f"Failed to get self correlation for {alpha_id} after {max_retries} attempts: {str(e)}", "ERROR")
                    raise
        
        # This should never be reached, but just in case
        return {}

    async def check_correlation(self, alpha_id: str, correlation_type: str = "both", threshold: float = 0.7) -> Dict[str, Any]:
        """Check alpha correlation against production alphas, self alphas, or both."""
        await self.ensure_authenticated()
        
        try:
            results = {
                'alpha_id': alpha_id,
                'threshold': threshold,
                'correlation_type': correlation_type,
                'checks': {}
            }
            
            # Determine which correlations to check
            check_types = []
            if correlation_type == "both":
                check_types = ["production", "self"]
            else:
                check_types = [correlation_type]
            
            all_passed = True
            
            for check_type in check_types:
                if check_type == "production":
                    correlation_data = await self.get_production_correlation(alpha_id)
                elif check_type == "self":
                    correlation_data = await self.get_self_correlation(alpha_id)
                else:
                    continue
                
                # Analyze correlation data (robust to schema/records format)
                if isinstance(correlation_data, dict):
                    # Prefer strict access to schema.max or top-level max; otherwise error
                    schema = correlation_data.get('schema') or {}
                    if isinstance(schema, dict) and 'max' in schema:
                        max_correlation = float(schema['max'])
                    elif 'max' in correlation_data:
                        # Some endpoints place max at top-level
                        max_correlation = float(correlation_data['max'])
                    else:
                        # Attempt to derive from records; if none found, raise error instead of defaulting
                        records = correlation_data.get('records') or []
                        if isinstance(records, list) and records:
                            candidate_max = None
                            for row in records:
                                if isinstance(row, (list, tuple)):
                                    for v in row:
                                        try:
                                            vf = float(v)
                                            if -1.0 <= vf <= 1.0:
                                                candidate_max = vf if candidate_max is None else max(candidate_max, vf)
                                        except Exception:
                                            continue
                                elif isinstance(row, dict):
                                    for key in ('correlation', 'prodCorrelation', 'selfCorrelation', 'max'):
                                        try:
                                            vf = float(row.get(key))
                                            if -1.0 <= vf <= 1.0:
                                                candidate_max = vf if candidate_max is None else max(candidate_max, vf)
                                        except Exception:
                                            continue
                            if candidate_max is None:
                                raise ValueError("Unable to derive max correlation from records")
                            max_correlation = float(candidate_max)
                        else:
                            raise KeyError("Correlation response missing 'schema.max' or top-level 'max' and no 'records' to derive from")
                else:
                    raise TypeError("Correlation data is not a dictionary")

                passes_check = max_correlation < threshold
                
                results['checks'][check_type] = {
                    'max_correlation': max_correlation,
                    'passes_check': passes_check,
                    'correlation_data': correlation_data
                }
                
                if not passes_check:
                    all_passed = False
            
            results['all_passed'] = all_passed
            
            return results
            
        except Exception as e:
            self.log(f"Failed to check correlation: {str(e)}", "ERROR")
            raise

    async def get_submission_check(self, alpha_id: str) -> Dict[str, Any]:
        """Comprehensive pre-submission check."""
        await self.ensure_authenticated()
        
        try:
            # Get correlation checks using the unified function
            correlation_checks = await self.check_correlation(alpha_id, correlation_type="both")
            
            # Get alpha details for additional validation
            alpha_details = await self.get_alpha_details(alpha_id)
            
            # Compile comprehensive check results
            checks = {
                'correlation_checks': correlation_checks,
                'alpha_details': alpha_details,
                'all_passed': correlation_checks['all_passed']
            }
            
            return checks
        except Exception as e:
            self.log(f"Failed to get submission check: {str(e)}", "ERROR")
            raise

    async def set_alpha_properties(
        self,
        alpha_id: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        category: Optional[str] = None,
        regular_desc: Optional[str] = None,
        selection_desc: Optional[str] = None,
        combo_desc: Optional[str] = None,
        osmosis_points: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update alpha properties (name, color, tags, descriptions, category, osmosis points)."""
        await self.ensure_authenticated()

        try:
            if osmosis_points is not None and not (1 <= osmosis_points <= 100000):
                raise ValueError(f"osmosis_points must be between 1 and 100000, got {osmosis_points}")

            option_map = {
                "name": name,
                "color": color,
                "tags": tags,
                "category": category,
                "osmosisPoints": osmosis_points,
                "regular": {"description": regular_desc} if regular_desc is not None else None,
                "selection": {"description": selection_desc} if selection_desc is not None else None,
                "combo": {"description": combo_desc} if combo_desc is not None else None,
            }
            data = {k: v for k, v in option_map.items() if v is not None}

            response = await self._request('patch', f"{self.base_url}/alphas/{alpha_id}", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to set alpha properties: {str(e)}", "ERROR")
            raise

    async def get_record_sets(self, alpha_id: str) -> Dict[str, Any]:
        """List available record sets for an alpha."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/recordsets")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get record sets: {str(e)}", "ERROR")
            raise

    async def get_record_set_data(self, alpha_id: str, record_set_name: str) -> Dict[str, Any]:
        """Get data from a specific record set."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/recordsets/{record_set_name}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get record set data: {str(e)}", "ERROR")
            raise

    async def get_user_activities(self, user_id: str, grouping: Optional[str] = None) -> Dict[str, Any]:
        """Get user activity diversity data."""
        await self.ensure_authenticated()
        
        try:
            params = {}
            if grouping:
                params['grouping'] = grouping
            
            response = await self._request('get', f"{self.base_url}/users/{user_id}/activities", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get user activities: {str(e)}", "ERROR")
            raise

    async def get_pyramid_multipliers(self) -> Dict[str, Any]:
        """Get current pyramid multipliers showing BRAIN's encouragement levels."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/users/self/activities/pyramid-multipliers")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get pyramid multipliers: {str(e)}", "ERROR")
            raise

    async def get_pyramid_alphas(self, start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get user's current alpha distribution across pyramid categories."""
        await self.ensure_authenticated()
        
        try:
            params = {}
            if start_date:
                params['startDate'] = start_date
            if end_date:
                params['endDate'] = end_date
            
            # Try the user-specific activities endpoint first (like pyramid-multipliers)
            response = await self._request('get', f"{self.base_url}/users/self/activities/pyramid-alphas", params=params)

            # If that fails, try alternative endpoints
            if response.status_code == 404:
                # Try alternative endpoint structure
                response = await self._request('get', f"{self.base_url}/users/self/pyramid/alphas", params=params)

                if response.status_code == 404:
                    # Try yet another alternative
                    response = await self._request('get', f"{self.base_url}/activities/pyramid-alphas", params=params)
                    
                    if response.status_code == 404:
                        # Return an informative error with what we tried
                        return {
                            "error": "Pyramid alphas endpoint not found",
                            "tried_endpoints": [
                                "/users/self/activities/pyramid-alphas",
                                "/users/self/pyramid/alphas", 
                                "/activities/pyramid-alphas",
                                "/pyramid/alphas"
                            ],
                            "suggestion": "This endpoint may not be available in the current API version"
                        }
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get pyramid alphas: {str(e)}", "ERROR")
            raise
            
    async def get_user_competitions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get list of competitions that the user is participating in."""
        await self.ensure_authenticated()
        
        try:
            if not user_id:
                # Get current user ID if not specified
                user_response = await self._request('get', f"{self.base_url}/users/self")
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    user_id = user_data.get('id')
                else:
                    user_id = 'self'

            response = await self._request('get', f"{self.base_url}/users/{user_id}/competitions")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get user competitions: {str(e)}", "ERROR")
            raise
            
    async def get_competition_details(self, competition_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific competition."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/competitions/{competition_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get competition details: {str(e)}", "ERROR")
            raise
            
    async def get_competition_agreement(self, competition_id: str) -> Dict[str, Any]:
        """Get the rules, terms, and agreement for a specific competition."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/competitions/{competition_id}/agreement")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get competition agreement: {str(e)}", "ERROR")
            raise

    async def get_platform_setting_options(self) -> Dict[str, Any]:
        """Get available instrument types, regions, delays, and universes."""
        await self.ensure_authenticated()
        
        try:
            # Use OPTIONS method on simulations endpoint to get configuration options
            response = await self._request('options', f"{self.base_url}/simulations")
            response.raise_for_status()
            
            # Parse the settings structure from the response
            settings_data = response.json()
            settings_options = settings_data['actions']['POST']['settings']['children']
            
            # Extract instrument configuration options
            instrument_type_data = {}
            region_data = {}
            universe_data = {}
            delay_data = {}
            neutralization_data = {}
            
            # Parse each setting type
            for key, setting in settings_options.items():
                if setting['type'] == 'choice':
                    if setting['label'] == 'Instrument type':
                        instrument_type_data = setting['choices']
                    elif setting['label'] == 'Region':
                        region_data = setting['choices']['instrumentType']
                    elif setting['label'] == 'Universe':
                        universe_data = setting['choices']['instrumentType']
                    elif setting['label'] == 'Delay':
                        delay_data = setting['choices']['instrumentType']
                    elif setting['label'] == 'Neutralization':
                        neutralization_data = setting['choices']['instrumentType']
            
            # Build comprehensive instrument options
            data_list = []
            
            for instrument_type in instrument_type_data:
                for region in region_data[instrument_type['value']]:
                    for delay in delay_data[instrument_type['value']]['region'][region['value']]:
                        row = {
                            'InstrumentType': instrument_type['value'],
                            'Region': region['value'],
                            'Delay': delay['value']
                        }
                        row['Universe'] = [
                            item['value'] for item in universe_data[instrument_type['value']]['region'][region['value']]
                        ]
                        row['Neutralization'] = [
                            item['value'] for item in neutralization_data[instrument_type['value']]['region'][region['value']]
                        ]
                        data_list.append(row)
            
            # Return structured data
            return {
                'instrument_options': data_list,
                'total_combinations': len(data_list),
                'instrument_types': [item['value'] for item in instrument_type_data],
                'regions_by_type': {
                    item['value']: [r['value'] for r in region_data[item['value']]]
                    for item in instrument_type_data
                }
            }
            
        except Exception as e:
            self.log(f"Failed to get instrument options: {str(e)}", "ERROR")
            raise
            
    async def performance_comparison(self, alpha_id: str, team_id: Optional[str] = None, 
                                     competition: Optional[str] = None) -> Dict[str, Any]:
        """Get performance comparison data for an alpha."""
        await self.ensure_authenticated()
        
        try:
            params = {"teamId": team_id, "competition": competition}
            params = {k: v for k, v in params.items() if v is not None}
            
            response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/performance-comparison", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get performance comparison: {str(e)}", "ERROR")
            raise
            
    # --- Helper function for data flattening ---
    
    async def expand_nested_data(self, data: List[Dict[str, Any]], preserve_original: bool = True) -> List[Dict[str, Any]]:
        """Flatten complex nested data structures into tabular format."""
        try:
            df = pd.json_normalize(data, sep='_')
            if preserve_original:
                original_df = pd.DataFrame(data)
                df = pd.concat([original_df, df], axis=1)
                df = df.loc[:,~df.columns.duplicated()]
            return df.to_dict(orient='records')
        except Exception as e:
            self.log(f"Failed to expand nested data: {str(e)}", "ERROR")
            raise
            
    # --- New documentation endpoint ---
    
    async def get_documentation_page(self, page_id: str) -> Dict[str, Any]:
        """Retrieve detailed content of a specific documentation page/article."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/tutorial-pages/{page_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get documentation page: {str(e)}", "ERROR")
            raise

brain_client = BrainApiClient()

# --- Configuration Management ---

def _resolve_config_path(for_write: bool = False) -> str:
    """
    Resolve the configuration file path.
    
    Checks for a file specified by the MCP_CONFIG_FILE environment variable,
    then falls back to ~/.brain_mcp_config.json. If for_write is True,
    it ensures the directory exists.
    """
    if 'MCP_CONFIG_FILE' in os.environ:
        return os.environ['MCP_CONFIG_FILE']
    
    config_path = Path(__file__).parent / "user_config.json"
    
    if for_write:
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
        except (IOError, OSError) as e:
            logger.warning(f"Could not create config directory {config_path.parent}: {e}")
            # Fallback to a temporary file if home is not writable
            import tempfile
            return tempfile.NamedTemporaryFile(delete=False).name
            
    return str(config_path)

def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    config_file = _resolve_config_path()
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config file {config_file}: {e}")
    return {}

def save_config(config: Dict[str, Any]):
    """Save configuration to file using the resolved config path.
    
    This function now uses the write-enabled path resolver to handle
    cases where the default home directory is not writable.
    """
    config_file = _resolve_config_path(for_write=True)
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving config file to {config_file}: {e}")

# --- MCP Tool Definitions ---

mcp = FastMCP(
    "brain-platform-mcp",
    "A server for interacting with the WorldQuant BRAIN platform",
)

@mcp.tool()
async def authenticate(email: Optional[str] = "", password: Optional[str] = "") -> Dict[str, Any]:
    """
    🔐 Authenticate with WorldQuant BRAIN platform.
    
    This is the first step in any BRAIN workflow. You must authenticate before using any other tools.
    
    Args:
        email: Your BRAIN platform email address (optional if in config or .brain_credentials)
        password: Your BRAIN platform password (optional if in config or .brain_credentials)
    
    Returns:
        Authentication result with user info and permissions
    """
    try:
        # Load config to get credentials if not provided
        config = load_config()
        credentials = config.get("credentials", {})
        email = email or credentials.get("email")
        password = password or credentials.get("password")
        if not email or not password:
            return {"error": "Authentication credentials not provided or found in config."}
        
        auth_result = await brain_client.authenticate(email, password)
        
        # Save successful credentials
        if auth_result.get('status') == 'authenticated':
            if 'credentials' not in config:
                config['credentials'] = {}
            config['credentials']['email'] = email
            config['credentials']['password'] = password
            save_config(config)
            
        return auth_result
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def manage_config(action: str = "get", settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    🔧 Manage configuration settings - get or update configuration.
    
    Args:
        action: Action to perform ("get" to retrieve config, "set" to update config)
        settings: Configuration settings to update (required when action="set")
    
    Returns:
        Current or updated configuration including authentication status
    """
    if action == "get":
        config = load_config()
        auth_status = await brain_client.get_authentication_status()
        
        return {
            "config": config,
            "auth_status": auth_status,
            "is_authenticated": await brain_client.is_authenticated()
        }
    
    elif action == "set":
        if settings is None:
            return {"error": "Settings parameter is required when action='set'"}
        
        config = load_config()
        config.update(settings)
        save_config(config)
        return config
    
    else:
        return {"error": f"Invalid action '{action}'. Use 'get' or 'set'."}

# --- Simulation Tools ---

@mcp.tool()
async def create_simulation(
    type: str = "REGULAR",
    instrument_type: str = "EQUITY",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: float = 0.0,
    neutralization: str = "NONE",
    truncation: float = 0.0,
    test_period: str = "P0Y0M",
    unit_handling: str = "VERIFY",
    nan_handling: str = "OFF",
    language: str = "FASTEXPR",
    visualization: bool = True,
    regular: Optional[str] = None,
    combo: Optional[str] = None,
    selection: Optional[str] = None,
    pasteurization: str = "ON",
    max_trade: str = "OFF",
    selection_handling: str = "POSITIVE",
    selection_limit: int = 1000,
    component_activation: str = "IS",
    max_position: str = "OFF",
) -> Dict[str, Any]:
    """
    🚀 Create a new simulation on BRAIN platform.
    
    This tool creates and starts a simulation with your alpha code. Use this after you have your alpha formula ready.
    
    Args:
        type: Simulation type ("REGULAR" or "SUPER")
        instrument_type: Type of instruments (e.g., "EQUITY")
        region: Market region (e.g., "USA")
        universe: Universe of stocks (e.g., "TOP3000")
        delay: Data delay (0 or 1)
        decay: Decay value for the simulation
        neutralization: Neutralization method
        truncation: Truncation value
        test_period: Test period (e.g., "P0Y0M" for 1 year 6 months)
        unit_handling: Unit handling method
        nan_handling: NaN handling method
        language: Expression language (e.g., "FASTEXPR")
        visualization: Enable visualization
        regular: Regular simulation code (for REGULAR type)
        combo: Combo code (for SUPER type)
        selection: Selection code (for SUPER type)
    
    Returns:
        Simulation creation result with ID and location
    """
    try:
        settings = SimulationSettings(
            instrumentType=instrument_type,
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            testPeriod=test_period,
            unitHandling=unit_handling,
            nanHandling=nan_handling,
            language=language,
            visualization=visualization,
            pasteurization=pasteurization,
            maxTrade=max_trade,
            selectionHandling=selection_handling,
            selectionLimit=selection_limit,
            componentActivation=component_activation,
            maxPosition=max_position,
        )
        
        sim_data = SimulationData(
            type=type,
            settings=settings,
            regular=regular,
            combo=combo,
            selection=selection
        )
        
        return await brain_client.create_simulation(sim_data)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Alpha and Data Retrieval Tools ---

@mcp.tool()
async def get_alpha_details(alpha_id: str) -> Dict[str, Any]:
    """
    📋 Get detailed information about an alpha.
    
    Args:
        alpha_id: The ID of the alpha to retrieve
    
    Returns:
        Detailed alpha information
    """
    try:
        return await brain_client.get_alpha_details(alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_datasets(
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: str = "false",
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """
    📚 Get available datasets for research.
    
    Use this to discover what data is available for your alpha research.
    
    Args:
        instrument_type: Type of instruments (e.g., "EQUITY")
        region: Market region (e.g., "USA")
        delay: Data delay (0 or 1)
        universe: Universe of stocks (e.g., "TOP3000")
        theme: Theme filter
    
    Returns:
        Available datasets
    """
    try:
        return await brain_client.get_datasets(instrument_type, region, delay, universe, theme, search)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_datafields(
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: str = "false",
    dataset_id: Optional[str] = None,
    data_type: str = "",
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """
    🔍 Get available data fields for alpha construction.
    
    Use this to find specific data fields you can use in your alpha formulas.
    
    Args:
        instrument_type: Type of instruments (e.g., "EQUITY")
        region: Market region (e.g., "USA")
        delay: Data delay (0 or 1)
        universe: Universe of stocks (e.g., "TOP3000")
        theme: Theme filter
        dataset_id: Specific dataset ID to filter by
        data_type: Type of data (e.g., "MATRIX",'VECTOR','GROUP')
        search: Search term to filter fields
    
    Returns:
        Available data fields
    """
    try:
        return await brain_client.get_datafields(instrument_type, region, delay, universe, theme, dataset_id, data_type, search)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_alpha_pnl(alpha_id: str) -> Dict[str, Any]:
    """
    📈 Get PnL (Profit and Loss) data for an alpha.
    
    Args:
        alpha_id: The ID of the alpha
    
    Returns:
        PnL data for the alpha
    """
    try:
        return await brain_client.get_alpha_pnl(alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_user_alphas(
    stage: str = "IS",
    limit: int = 30,
    offset: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    submission_start_date: Optional[str] = None,
    submission_end_date: Optional[str] = None,
    order: Optional[str] = None,
    hidden: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    👤 Get user's alphas with advanced filtering, pagination, and sorting.

    This tool retrieves a list of your alphas, allowing for detailed filtering based on stage,
    creation date, submission date, and visibility. It also supports pagination and custom sorting.

    Args:
        stage (str): The stage of the alphas to retrieve.
            - "IS": In-Sample (alphas that have not been submitted).
            - "OS": Out-of-Sample (alphas that have been submitted).
            Defaults to "IS".
        limit (int): The maximum number of alphas to return in a single request.
            For example, `limit=50` will return at most 50 alphas. Defaults to 30.
        offset (int): The number of alphas to skip from the beginning of the list.
            Used for pagination. For example, `limit=50, offset=50` will retrieve alphas 51-100.
            Defaults to 0.
        start_date (Optional[str]): The earliest creation date for the alphas to be included.
            Filters for alphas created on or after this date.
            Example format: "2023-01-01T00:00:00Z".
        end_date (Optional[str]): The latest creation date for the alphas to be included.
            Filters for alphas created before this date.
            Example format: "2023-12-31T23:59:59Z".
        submission_start_date (Optional[str]): The earliest submission date for the alphas.
            Only applies to "OS" alphas. Filters for alphas submitted on or after this date.
            Example format: "2024-01-01T00:00:00Z".
        submission_end_date (Optional[str]): The latest submission date for the alphas.
            Only applies to "OS" alphas. Filters for alphas submitted before this date.
            Example format: "2024-06-30T23:59:59Z".
        order (Optional[str]): The sorting order for the returned alphas.
            Prefix with a hyphen (-) for descending order.
            Examples: "name" (sort by name ascending), "-dateSubmitted" (sort by submission date descending).
        hidden (Optional[bool]): Filter alphas based on their visibility.
            - `True`: Only return hidden alphas.
            - `False`: Only return non-hidden alphas.
            If not provided, both hidden and non-hidden alphas are returned.

    Returns:
        Dict[str, Any]: A dictionary containing a list of alpha details under the 'results' key,
        along with pagination information. If an error occurs, it returns a dictionary with an 'error' key.
    """
    try:
        return await brain_client.get_user_alphas(
            stage=stage, limit=limit, offset=offset, start_date=start_date,
            end_date=end_date, submission_start_date=submission_start_date,
            submission_end_date=submission_end_date, order=order, hidden=hidden
        )
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def submit_alpha(alpha_id: str) -> Dict[str, Any]:
    """
    📤 Submit an alpha for production.
    
    Use this when your alpha is ready for production deployment.
    
    Args:
        alpha_id: The ID of the alpha to submit
    
    Returns:
        Submission result
    """
    try:
        success = await brain_client.submit_alpha(alpha_id)
        return {"success": success}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def value_factor_trendScore(start_date: str, end_date: str) -> Dict[str, Any]:
    """Compute and return the diversity score for REGULAR alphas in a submission-date window.
    This function calculate the diversity of the users' submission, by checking the diversity, we can have a good understanding on the valuefactor's trend.
    This MCP tool wraps BrainApiClient.value_factor_trendScore and always uses submission dates (OS).

    Inputs:
        - start_date: ISO UTC start datetime (e.g. '2025-08-14T00:00:00Z')
        - end_date: ISO UTC end datetime (e.g. '2025-08-18T23:59:59Z')
        - p_max: optional integer total number of pyramid categories for normalization

    Returns: compact JSON with diversity_score, N, A, P, P_max, S_A, S_P, S_H, per_pyramid_counts
    """
    try:
        return await brain_client.value_factor_trendScore(start_date=start_date, end_date=end_date)
    except Exception as e:
        return {"error": str(e)}

# --- Community and Events Tools ---

@mcp.tool()
async def get_events() -> Dict[str, Any]:
    """
    🏆 Get available events and competitions.
    
    Returns:
        Available events and competitions
    """
    try:
        return await brain_client.get_events()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_leaderboard(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    🏅 Get leaderboard data.
    
    Args:
        user_id: Optional user ID to filter results
    
    Returns:
        Leaderboard data
    """
    try:
        return await brain_client.get_leaderboard(user_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


# --- Forum Tools ---

@mcp.tool()
async def get_operators() -> Dict[str, Any]:
    """
    🔧 Get available operators for alpha creation.
    
    Returns:
        Dictionary containing operators list and count
    """
    try:
        operators = await brain_client.get_operators()
        if isinstance(operators, list):
            return {"results": operators, "count": len(operators)}
        return operators
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def run_selection(
    selection: str,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    selection_limit: int = 1000,
    selection_handling: str = "POSITIVE",
) -> Dict[str, Any]:
    """
    🎯 Run a selection query to filter instruments.
    
    Args:
        selection: Selection criteria
        instrument_type: Type of instruments
        region: Geographic region
        delay: Delay setting
        selection_limit: Maximum number of results
        selection_handling: How to handle selection results
    
    Returns:
        Selection results
    """
    try:
        return await brain_client.run_selection(
            selection, instrument_type, region, delay, selection_limit, selection_handling
        )
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_user_profile(user_id: str = "self") -> Dict[str, Any]:
    """
    👤 Get user profile information.
    
    Args:
        user_id: User ID (default: "self" for current user)
    
    Returns:
        User profile data
    """
    try:
        return await brain_client.get_user_profile(user_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_documentations() -> Dict[str, Any]:
    """
    📚 Get available documentations and learning materials.
    
    Returns:
        List of documentations
    """
    try:
        return await brain_client.get_documentations()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Message and Forum Tools ---

@mcp.tool()
async def get_messages(limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
    """
    💬 Get messages for the current user with optional pagination.
    
    Args:
        limit: Maximum number of messages to return (e.g., 10 for top 10 messages)
        offset: Number of messages to skip (for pagination)
    
    Returns:
        Messages for the current user, optionally limited by count
    """
    try:
        return await brain_client.get_messages(limit, offset)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_glossary_terms(email: str = "", password: str = "") -> List[Dict[str, str]]:
    """
    📚 Get glossary terms from WorldQuant BRAIN forum.
    
    Note: This uses Playwright and is implemented in forum_functions.py
    
    Args:
        email: Your BRAIN platform email address (optional if in config)
        password: Your BRAIN platform password (optional if in config)
    
    Returns:
        A list of glossary terms with definitions
    """
    try:
        config = load_config()
        credentials = config.get("credentials", {})
        email = email or credentials.get("email")
        password = password or credentials.get("password")
        if not email or not password:
            raise ValueError("Authentication credentials not provided or found in config.")
        
        return await brain_client.get_glossary_terms(email, password)
    except Exception as e:
        logger.error(f"Error in get_glossary_terms tool: {e}")
        return [{"error": str(e)}]

@mcp.tool()
async def search_forum_posts(search_query: str, email: str = "", password: str = "", 
                             max_results: int = 50) -> Dict[str, Any]:
    """
    🔍 Search forum posts on WorldQuant BRAIN support site.
    
    Note: This uses Playwright and is implemented in forum_functions.py
    
    Args:
        search_query: Search term or phrase
        email: Your BRAIN platform email address (optional if in config)
        password: Your BRAIN platform password (optional if in config)
        max_results: Maximum number of results to return (default: 50)
    
    Returns:
        Search results with analysis
    """
    try:
        config = load_config()
        credentials = config.get("credentials", {})
        email = email or credentials.get("email")
        password = password or credentials.get("password")
        if not email or not password:
            return {"error": "Authentication credentials not provided or found in config."}
            
        return await brain_client.search_forum_posts(email, password, search_query, max_results)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def read_forum_post(article_id: str, email: str = "", password: str = "", 
                          include_comments: bool = True) -> Dict[str, Any]:
    """
    📄 Get a specific forum post by article ID.
    
    Note: This uses Playwright and is implemented in forum_functions.py
    
    Args:
        article_id: The article ID to retrieve (e.g., "32984819083415-新人求模板")
        email: Your BRAIN platform email address (optional if in config)
        password: Your BRAIN platform password (optional if in config)
    
    Returns:
        Forum post content with comments
    """
    try:
        config = load_config()
        credentials = config.get("credentials", {})
        email = email or credentials.get("email")
        password = password or credentials.get("password")
        if not email or not password:
            return {"error": "Authentication credentials not provided or found in config."}

        return await brain_client.read_forum_post(email, password, article_id, include_comments)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_alpha_yearly_stats(alpha_id: str) -> Dict[str, Any]:
    """Get yearly statistics for an alpha."""
    try:
        return await brain_client.get_alpha_yearly_stats(alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def check_correlation(alpha_id: str, correlation_type: str = "both", threshold: float = 0.7) -> Dict[str, Any]:
    """Check alpha correlation against production alphas, self alphas, or both."""
    try:
        return await brain_client.check_correlation(alpha_id, correlation_type, threshold)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_submission_check(alpha_id: str) -> Dict[str, Any]:
    """Comprehensive pre-submission check."""
    try:
        return await brain_client.get_submission_check(alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def set_alpha_properties(
    alpha_id: str,
    name: Optional[str] = None,
    color: Optional[str] = None,
    category: Optional[str] = None,
    regular_desc: Optional[str] = None,
    selection_desc: Optional[str] = None,
    combo_desc: Optional[str] = None,
    osmosis_points: Optional[int] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Update alpha properties (name, color, tags, descriptions, category, osmosis points)."""
    try:
        return await brain_client.set_alpha_properties(
            alpha_id,
            name,
            color,
            category,
            regular_desc,
            selection_desc,
            combo_desc,
            osmosis_points,
            tags,
        )
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_record_sets(alpha_id: str) -> Dict[str, Any]:
    """List available record sets for an alpha."""
    try:
        return await brain_client.get_record_sets(alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_record_set_data(alpha_id: str, record_set_name: str) -> Dict[str, Any]:
    """Get data from a specific record set."""
    try:
        return await brain_client.get_record_set_data(alpha_id, record_set_name)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_user_activities(user_id: str, grouping: Optional[str] = None) -> Dict[str, Any]:
    """Get user activity diversity data."""
    try:
        return await brain_client.get_user_activities(user_id, grouping)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_pyramid_multipliers() -> Dict[str, Any]:
    """Get current pyramid multipliers showing BRAIN's encouragement levels."""
    try:
        return await brain_client.get_pyramid_multipliers()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_pyramid_alphas(start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
    """Get user's current alpha distribution across pyramid categories."""
    try:
        return await brain_client.get_pyramid_alphas(start_date, end_date)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
        
@mcp.tool()
async def get_user_competitions(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get list of competitions that the user is participating in."""
    try:
        return await brain_client.get_user_competitions(user_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_competition_details(competition_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific competition."""
    try:
        return await brain_client.get_competition_details(competition_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_competition_agreement(competition_id: str) -> Dict[str, Any]:
    """Get the rules, terms, and agreement for a specific competition."""
    try:
        return await brain_client.get_competition_agreement(competition_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_platform_setting_options() -> Dict[str, Any]:
    """Discover valid simulation setting options (instrument types, regions, delays, universes, neutralization).

    Use this when a simulation request might contain an invalid/mismatched setting. If an AI or user supplies
    incorrect parameters (e.g., wrong region for an instrument type), call this tool to retrieve the authoritative
    option sets and correct the inputs before proceeding.

    Returns:
        A structured list of valid combinations and choice lists to validate or fix simulation settings.
    """
    try:
        return await brain_client.get_platform_setting_options()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def performance_comparison(alpha_id: str, team_id: Optional[str] = None, 
                                 competition: Optional[str] = None) -> Dict[str, Any]:
    """Get performance comparison data for an alpha."""
    try:
        return await brain_client.performance_comparison(alpha_id, team_id, competition)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
        
# --- Dataframe Tool ---

@mcp.tool()
async def expand_nested_data(data: List[Dict[str, Any]], preserve_original: bool = True) -> List[Dict[str, Any]]:
    """Flatten complex nested data structures into tabular format."""
    try:
        return await brain_client.expand_nested_data(data, preserve_original)
    except Exception as e:
        return [{"error": f"An unexpected error occurred: {str(e)}"}]
        
# --- Documentation Tool ---

@mcp.tool()
async def get_documentation_page(page_id: str) -> Dict[str, Any]:
    """Retrieve detailed content of a specific documentation page/article."""
    try:
        return await brain_client.get_documentation_page(page_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Advanced Simulation Tools ---

@mcp.tool()
async def create_multi_simulation(
    alpha_expressions: List[str],
    instrument_type: str = "EQUITY",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: float = 0.0,
    neutralization: str = "NONE",
    truncation: float = 0.0,
    test_period: str = "P0Y0M",
    unit_handling: str = "VERIFY",
    nan_handling: str = "OFF",
    language: str = "FASTEXPR",
    visualization: bool = True,
    pasteurization: str = "ON",
    max_trade: str = "OFF"
) -> Dict[str, Any]:
    """
    🚀 Create multiple regular alpha simulations on BRAIN platform in a single request.
    
    This tool creates a multisimulation with multiple regular alpha expressions,
    waits for all simulations to complete, and returns detailed results for each alpha.
    
    ⏰ NOTE: Multisimulations can take 8+ minutes to complete. This tool will wait
    for the entire process and return comprehensive results.
    Call get_platform_setting_options to get the valid options for the simulation.
    Args:
        alpha_expressions: List of alpha expressions (2-8 expressions required)
        instrument_type: Type of instruments (default: "EQUITY")
        region: Market region (default: "USA")
        universe: Universe of stocks (default: "TOP3000")
        delay: Data delay (default: 1)
        decay: Decay value (default: 0.0)
        neutralization: Neutralization method (default: "NONE")
        truncation: Truncation value (default: 0.0)
        test_period: Test period (default: "P0Y0M")
        unit_handling: Unit handling method (default: "VERIFY")
        nan_handling: NaN handling method (default: "OFF")
        language: Expression language (default: "FASTEXPR")
        visualization: Enable visualization (default: True)
        pasteurization: Pasteurization setting (default: "ON")
        max_trade: Max trade setting (default: "OFF")
    
    Returns:
        Dictionary containing multisimulation results and individual alpha details
    """
    try:
        # Validate input
        if len(alpha_expressions) < 2:
            return {"error": "At least 2 alpha expressions are required"}
        if len(alpha_expressions) > 8:
            return {"error": "Maximum 8 alpha expressions allowed per request"}
        
        # Create multisimulation data
        multisimulation_data = []
        for alpha_expr in alpha_expressions:
            simulation_item = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': instrument_type,
                    'region': region,
                    'universe': universe,
                    'delay': delay,
                    'decay': decay,
                    'neutralization': neutralization,
                    'truncation': truncation,
                    'pasteurization': pasteurization,
                    'unitHandling': unit_handling,
                    'nanHandling': nan_handling,
                    'language': language,
                    'visualization': visualization,
                    'testPeriod': test_period,
                    'maxTrade': max_trade
                },
                'regular': alpha_expr
            }
            multisimulation_data.append(simulation_item)
        
        # Send multisimulation request
        response = brain_client.session.post(f"{brain_client.base_url}/simulations", json=multisimulation_data)
        
        if response.status_code != 201:
            return {"error": f"Failed to create multisimulation. Status: {response.status_code}"}
        
        # Get multisimulation location
        location = response.headers.get('Location', '')
        if not location:
            return {"error": "No location header in multisimulation response"}
        
        # Wait for children to appear and get results
        return await _wait_for_multisimulation_completion(location, len(alpha_expressions))
        
    except Exception as e:
        return {"error": f"Error creating multisimulation: {str(e)}"}

async def _wait_for_multisimulation_completion(location: str, expected_children: int) -> Dict[str, Any]:
    """Wait for multisimulation to complete and return results"""
    try:
        # Simple progress indicator for users
        print(f"Waiting for multisimulation to complete... (this may take several minutes)")
        print(f"Expected {expected_children} alpha simulations")
        print()
        # Wait for children to appear - much more tolerant for 8+ minute multisimulations
        children = []
        max_wait_attempts = 200  # Increased significantly for 8+ minute multisimulations
        wait_attempt = 0
        
        while wait_attempt < max_wait_attempts and len(children) == 0:
            wait_attempt += 1
            
            try:
                multisim_response = brain_client.session.get(location)
                if multisim_response.status_code == 200:
                    multisim_data = multisim_response.json()
                    children = multisim_data.get('children', [])
                    
                    if children:
                        break
                    else:
                        # Wait before next attempt - use longer intervals for multisimulations
                        retry_after = multisim_response.headers.get("Retry-After", 5)
                        wait_time = float(retry_after)
                        await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(5)
            except Exception as e:
                await asyncio.sleep(5)
        
        if not children:
            return {"error": f"Children did not appear within {max_wait_attempts} attempts (multisimulation may still be processing)"}
        
        # Process each child to get alpha results
        alpha_results = []
        for i, child_id in enumerate(children):
            try:
                # The children are full URLs, not just IDs
                child_url = child_id if child_id.startswith('http') else f"{brain_client.base_url}/simulations/{child_id}"
                
                # Wait for this alpha to complete - more tolerant timing
                finished = False
                max_alpha_attempts = 100  # Increased for longer alpha processing
                alpha_attempt = 0
                
                while not finished and alpha_attempt < max_alpha_attempts:
                    alpha_attempt += 1
                    
                    try:
                        alpha_progress = brain_client.session.get(child_url)
                        if alpha_progress.status_code == 200:
                            alpha_data = alpha_progress.json()
                            retry_after = alpha_progress.headers.get("Retry-After", 0)
                            
                            if retry_after == 0:
                                finished = True
                                break
                            else:
                                wait_time = float(retry_after)
                                await asyncio.sleep(wait_time)
                        else:
                            await asyncio.sleep(5)
                    except Exception as e:
                        await asyncio.sleep(5)
                
                if finished:
                    # Get alpha details from the completed simulation
                    alpha_id = alpha_data.get("alpha")
                    if alpha_id:
                        # Now get the actual alpha details from the alpha endpoint
                        alpha_details = brain_client.session.get(f"{brain_client.base_url}/alphas/{alpha_id}")
                        if alpha_details.status_code == 200:
                            alpha_detail_data = alpha_details.json()
                            alpha_results.append({
                                'alpha_id': alpha_id,
                                'location': child_url,
                                'details': alpha_detail_data
                            })
                        else:
                            alpha_results.append({
                                'alpha_id': alpha_id,
                                'location': child_url,
                                'error': f'Failed to get alpha details: {alpha_details.status_code}'
                            })
                    else:
                        alpha_results.append({
                            'location': child_url,
                            'error': 'No alpha ID found in completed simulation'
                        })
                else:
                    alpha_results.append({
                        'location': child_url,
                        'error': f'Alpha simulation did not complete within {max_alpha_attempts} attempts'
                    })
                    
            except Exception as e:
                alpha_results.append({
                    'location': f"child_{i+1}",
                    'error': str(e)
                })
        
        # Return comprehensive results
        print(f"Multisimulation completed! Retrieved {len(alpha_results)} alpha results")
        return {
            'success': True,
            'message': f'Successfully created {expected_children} regular alpha simulations',
            'total_requested': expected_children,
            'total_created': len(alpha_results),
            'multisimulation_id': location.split('/')[-1],
            'multisimulation_location': location,
            'alpha_results': alpha_results
        }
        
    except Exception as e:
        return {"error": f"Error waiting for multisimulation completion: {str(e)}"}
# --- Payment and Financial Tools ---

@mcp.tool()
async def get_daily_and_quarterly_payment(email: str = "", password: str = "") -> Dict[str, Any]:
    """
    Get daily and quarterly payment information from WorldQuant BRAIN platform.
    
    This function retrieves both base payments (daily alpha performance payments) and 
    other payments (competition rewards, quarterly payments, referrals, etc.).
    
    Args:
        email: Your BRAIN platform email address (optional if in config)
        password: Your BRAIN platform password (optional if in config)
    
    Returns:
        Dictionary containing base payment and other payment data with summaries and detailed records
    """
    try:
        config = load_config()
        credentials = config.get("credentials", {})
        email = email or credentials.get("email")
        password = password or credentials.get("password")
        if not email or not password:
            return {"error": "Authentication credentials not provided or found in config."}
            
        await brain_client.authenticate(email, password)
        
        # Get base payments
        try:
            base_response = brain_client.session.get(f"{brain_client.base_url}/users/self/activities/base-payment")
            base_response.raise_for_status()
            base_payments = base_response.json()
        except:
            base_payments = "no data"
            
        try:
            # Get other payments
            other_response = brain_client.session.get(f"{brain_client.base_url}/users/self/activities/other-payment")
            other_response.raise_for_status()
            other_payments = other_response.json()
        except:
            other_payments = "no data"    
        return {
            "base_payments": base_payments,
            "other_payments": other_payments
        }
        
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

from typing import Sequence
@mcp.tool()
async def lookINTO_SimError_message(locations: Sequence[str]) -> dict:
    """
    Fetch and parse error/status from multiple simulation locations (URLs).
    Args:
        locations: List of simulation result URLs (e.g., /simulations/{id})
    Returns:
        List of dicts with location, error message, and raw response
    """
    results = []
    for loc in locations:
        try:
            resp = brain_client.session.get(loc)
            if resp.status_code != 200:
                results.append({
                    "location": loc,
                    "error": f"HTTP {resp.status_code}",
                    "raw": resp.text
                })
                continue
            data = resp.json() if resp.text else {}
            # Try to extract error message or status
            error_msg = data.get("error") or data.get("message")
            # If alpha ID is missing, include that info
            if not data.get("alpha"):
                error_msg = error_msg or "Simulation did not get through, if you are running a multisimulation, check the other children location in your request"
            results.append({
                "location": loc,
                "error": error_msg,
                "raw": data
            })
        except Exception as e:
            results.append({
                "location": loc,
                "error": str(e),
                "raw": None
            })
    return {"results": results}


# --- Main entry point ---
if __name__ == "__main__":
    print("running the server")
    mcp.run()
