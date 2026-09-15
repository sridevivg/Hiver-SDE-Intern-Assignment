"""
SupportGraph AI — Canonical Troubleshooting Action Catalog & Semantic Aliasing.

Provides canonical troubleshooting sequences for operational problem families,
strict semantic equivalence mapping to prevent duplicate troubleshooting advice,
and safety/destructiveness gating.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Semantic equivalence mappings: canonical_name -> list of regex patterns / synonym phrases
SEMANTIC_ACTION_ALIASES: Dict[str, List[str]] = {
    "restart_device": [
        r"\b(restart|restarted|restarting)\b",
        r"\b(reboot|rebooted|rebooting)\b",
        r"\b(power\s*cycle[d]?)\b",
        r"\b(turn(?:ed)?\s*(?:(?:the\s*)?phone|device|it)?\s*off\s*and\s*(?:back\s*)?on)\b",
        r"\b(shut\s*down\s*and\s*(?:turn\s*on|restart))\b",
    ],
    "force_restart": [
        r"\b(force\s*restart(?:ed)?)\b",
        r"\b(hard\s*reboot(?:ed)?)\b",
        r"\b(hard\s*restart(?:ed)?)\b",
        r"\b(forced\s*restart)\b",
    ],
    "check_airplane_mode": [
        r"\b(airplane\s*mode)\b",
        r"\b(flight\s*mode)\b",
        r"\b(toggle[d]?\s*airplane)\b",
    ],
    "forget_and_reconnect_wifi": [
        r"\b(forget\s*(?:this\s*)?(?:wi-?fi|network))\b",
        r"\b(forgot\s*(?:the\s*)?(?:wi-?fi|network))\b",
        r"\b(reconnect(?:ed)?\s*to\s*(?:wi-?fi|network))\b",
        r"\b(re-?add\s*(?:wi-?fi|network))\b",
    ],
    "reset_network_settings": [
        r"\b(reset\s*(?:all\s*)?network\s*settings)\b",
        r"\b(network\s*reset)\b",
        r"\b(reset\s*network)\b",
    ],
    "toggle_bluetooth": [
        r"\b(toggle[d]?\s*bluetooth)\b",
        r"\b(turn(?:ed)?\s*bluetooth\s*off\s*and\s*(?:back\s*)?on)\b",
        r"\b(restart(?:ed)?\s*bluetooth)\b",
    ],
    "re_pair_bluetooth": [
        r"\b(un-?pair(?:ed)?\s*(?:and\s*re-?pair(?:ed)?)?)\b",
        r"\b(forget\s*(?:this\s*)?device)\b",
        r"\b(re-?pair(?:ed)?\s*(?:the\s*)?device)\b",
        r"\b(forgot\s*(?:bluetooth\s*)?device)\b",
    ],
    "test_alternative_charger": [
        r"\b(different\s*(?:charger|cable|cord|wall\s*adapter|plug|outlet))\b",
        r"\b(another\s*(?:charger|cable|cord|plug|outlet))\b",
        r"\b(tried\s*(?:a\s*)?(?:different|another)\s*(?:charger|cable|cord))\b",
        r"\b(changed\s*(?:the\s*)?(?:cable|cord|charger))\b",
    ],
    "clean_charging_port": [
        r"\b(clean(?:ed)?\s*(?:the\s*)?(?:charging\s*)?port)\b",
        r"\b(clear(?:ed)?\s*lint\s*(?:from\s*)?(?:the\s*)?port)\b",
        r"\b(blow\s*(?:into\s*)?(?:the\s*)?port)\b",
        r"\b(debris\s*in\s*port)\b",
    ],
    "clear_cache": [
        r"\b(clear(?:ed)?\s*(?:app\s*)?cache)\b",
        r"\b(wipe[d]?\s*cache)\b",
        r"\b(empty\s*cache)\b",
        r"\b(delete[d]?\s*cache)\b",
    ],
    "reinstall_app": [
        r"\b(reinstall(?:ed)?\s*(?:the\s*)?app)\b",
        r"\b(uninstall(?:ed)?\s*and\s*reinstall(?:ed)?)\b",
        r"\b(delete[d]?\s*and\s*reinstall(?:ed)?)\b",
        r"\b(fresh\s*install)\b",
        r"\b(remove[d]?\s*and\s*re-?add(?:ed)?)\b",
    ],
    "update_app": [
        r"\b(update[d]?\s*(?:the\s*)?app)\b",
        r"\b(app\s*update)\b",
        r"\b(installed\s*(?:the\s*)?update)\b",
        r"\b(update[d]?\s*to\s*latest\s*version)\b",
    ],
    "update_os": [
        r"\b(update[d]?\s*(?:ios|os|system|device|software))\b",
        r"\b(system\s*update)\b",
        r"\b(latest\s*(?:ios|os)\s*version)\b",
        r"\b(software\s*update)\b",
    ],
    "check_storage_space": [
        r"\b(check(?:ed)?\s*(?:iphone\s*)?storage)\b",
        r"\b(free(?:d)?\s*up\s*space)\b",
        r"\b(low\s*storage)\b",
        r"\b(manage\s*storage)\b",
        r"\b(out\s*of\s*storage)\b",
    ],
    "check_silent_mode": [
        r"\b(silent\s*(?:switch|mode))\b",
        r"\b(mute\s*(?:switch|mode))\b",
        r"\b(checked\s*mute)\b",
        r"\b(ring/?silent\s*switch)\b",
    ],
    "adjust_volume_settings": [
        r"\b(check(?:ed)?\s*volume)\b",
        r"\b(turned\s*up\s*volume)\b",
        r"\b(volume\s*slider)\b",
        r"\b(adjust(?:ed)?\s*volume)\b",
    ],
    "toggle_cellular_data": [
        r"\b(toggle[d]?\s*cellular)\b",
        r"\b(turn(?:ed)?\s*(?:off\s*and\s*on\s*)?mobile\s*data)\b",
        r"\b(cellular\s*data\s*toggle)\b",
    ],
    "reseat_sim_card": [
        r"\b(re-?seat(?:ed)?\s*sim)\b",
        r"\b(remove[d]?\s*(?:and\s*reinsert(?:ed)?)?\s*sim)\b",
        r"\b(take\s*out\s*sim)\b",
        r"\b(sim\s*tray)\b",
    ],
    "verify_apple_id": [
        r"\b(sign(?:ed)?\s*out\s*and\s*(?:back\s*)?in\s*(?:to\s*)?apple\s*id)\b",
        r"\b(verify\s*(?:apple\s*id|password|credentials))\b",
        r"\b(re-?enter(?:ed)?\s*password)\b",
        r"\b(check(?:ed)?\s*apple\s*id)\b",
    ],
    "check_spam_folder": [
        r"\b(check(?:ed)?\s*spam(?:\s*folder)?)\b",
        r"\b(check(?:ed)?\s*junk(?:\s*folder)?)\b",
        r"\b(looked\s*in\s*spam)\b",
        r"\b(spam\s*box)\b",
    ],
    "factory_reset": [
        r"\b(factory\s*reset)\b",
        r"\b(hard\s*reset)\b",
        r"\b(master\s*reset)\b",
        r"\b(restore\s*to\s*factory)\b",
        r"\b(erase\s*all\s*content)\b",
        r"\b(wiped?\s*device)\b",
    ],
}

# Canonical action metadata definitions
ACTION_METADATA: Dict[str, Dict[str, Any]] = {
    "restart_device": {
        "canonical_name": "restart_device",
        "description": "Power down device completely, wait 30 seconds, and turn back on.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Clears temporary volatile memory and resolves transient daemon errors.",
    },
    "force_restart": {
        "canonical_name": "force_restart",
        "description": "Perform hardware-level forced restart (Press Volume Up, then Volume Down, then hold Side button).",
        "is_destructive": False,
        "prerequisites": ["restart_device"],
        "expected_outcome": "Recovers unresponsive system processes and hard locks.",
    },
    "check_airplane_mode": {
        "canonical_name": "check_airplane_mode",
        "description": "Toggle Airplane Mode on for 15 seconds, then toggle off.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Forces cellular and wireless baseband chipsets to re-acquire cell towers and access points.",
    },
    "forget_and_reconnect_wifi": {
        "canonical_name": "forget_and_reconnect_wifi",
        "description": "Forget the Wi-Fi network in Settings > Wi-Fi, then rejoin with credentials.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Flushes stale DHCP leases and invalid authentication tokens.",
    },
    "reset_network_settings": {
        "canonical_name": "reset_network_settings",
        "description": "Reset Network Settings in Settings > General > Transfer or Reset > Reset.",
        "is_destructive": False,
        "prerequisites": ["forget_and_reconnect_wifi"],
        "expected_outcome": "Restores Wi-Fi networks, passwords, cellular settings, and VPN configurations to defaults.",
    },
    "toggle_bluetooth": {
        "canonical_name": "toggle_bluetooth",
        "description": "Toggle Bluetooth off in Settings > Bluetooth, wait 10 seconds, and turn back on.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Restarts Bluetooth stack and connection polling.",
    },
    "re_pair_bluetooth": {
        "canonical_name": "re_pair_bluetooth",
        "description": "Unpair the Bluetooth accessory ('Forget This Device') and re-pair from scratch.",
        "is_destructive": False,
        "prerequisites": ["toggle_bluetooth"],
        "expected_outcome": "Re-establishes fresh pairing keys and Bluetooth profile handshake.",
    },
    "test_alternative_charger": {
        "canonical_name": "test_alternative_charger",
        "description": "Test charging with a certified Apple Lightning/USB-C cable and a known-good wall outlet.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Isolates whether hardware fault is in the charging brick/cable or the device.",
    },
    "clean_charging_port": {
        "canonical_name": "clean_charging_port",
        "description": "Inspect and gently clean the charging port for lint or debris using an anti-static tool.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Ensures physical pin contacts meet cleanly without resistance.",
    },
    "clear_cache": {
        "canonical_name": "clear_cache",
        "description": "Clear app cached data or browser history and website data.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Purges corrupted cached assets causing rendering or execution crashes.",
    },
    "reinstall_app": {
        "canonical_name": "reinstall_app",
        "description": "Delete the affected application and reinstall the freshest build from the App Store.",
        "is_destructive": False,
        "prerequisites": ["clear_cache", "restart_device"],
        "expected_outcome": "Replaces corrupted sandbox binaries and database files.",
    },
    "update_app": {
        "canonical_name": "update_app",
        "description": "Open App Store, tap profile, and install any pending updates for the application.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Patches known software bugs and API compatibility issues.",
    },
    "update_os": {
        "canonical_name": "update_os",
        "description": "Navigate to Settings > General > Software Update and install latest iOS release.",
        "is_destructive": False,
        "prerequisites": ["restart_device"],
        "expected_outcome": "Provides system firmware patches and driver updates.",
    },
    "check_storage_space": {
        "canonical_name": "check_storage_space",
        "description": "Check Settings > General > iPhone Storage to confirm at least 10% free space.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Ensures virtual memory paging and system caches have required buffer space.",
    },
    "check_silent_mode": {
        "canonical_name": "check_silent_mode",
        "description": "Check the physical Ring/Silent switch on the side of the device.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Confirms hardware mute switch is not disengaged or orange-showing.",
    },
    "adjust_volume_settings": {
        "canonical_name": "adjust_volume_settings",
        "description": "Adjust volume controls in Settings > Sounds & Haptics and Control Center.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Rules out muted audio channel or zero volume level.",
    },
    "toggle_cellular_data": {
        "canonical_name": "toggle_cellular_data",
        "description": "Turn Cellular Data off and on in Settings > Cellular.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Forces baseband reconnection to carrier gateway.",
    },
    "reseat_sim_card": {
        "canonical_name": "reseat_sim_card",
        "description": "Eject the SIM tray, inspect the physical SIM for damage, and reinsert securely.",
        "is_destructive": False,
        "prerequisites": ["toggle_cellular_data"],
        "expected_outcome": "Resolves poor physical pin alignment on SIM card.",
    },
    "verify_apple_id": {
        "canonical_name": "verify_apple_id",
        "description": "Sign out of Apple ID in Settings and sign back in with primary credentials.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Renews expired authentication tokens and keychain access.",
    },
    "check_spam_folder": {
        "canonical_name": "check_spam_folder",
        "description": "Check email spam, junk, and trash folders for verification emails.",
        "is_destructive": False,
        "prerequisites": [],
        "expected_outcome": "Locates legitimate transactional emails routed away from primary inbox.",
    },
    "factory_reset": {
        "canonical_name": "factory_reset",
        "description": "Erase All Content and Settings (Settings > General > Transfer or Reset iPhone > Erase All Content and Settings).",
        "is_destructive": True,
        "prerequisites": ["force_restart", "update_os"],
        "expected_outcome": "Restores clean factory operating state. Deletes all user data.",
    },
}

# Progressive canonical action sequences per problem family
FAMILY_ACTION_SEQUENCES: Dict[str, List[str]] = {
    "CONNECTIVITY_WIFI": [
        "check_airplane_mode",
        "restart_device",
        "forget_and_reconnect_wifi",
        "reset_network_settings",
    ],
    "CONNECTIVITY_BLUETOOTH": [
        "toggle_bluetooth",
        "restart_device",
        "re_pair_bluetooth",
        "reset_network_settings",
    ],
    "NETWORK_CELLULAR": [
        "check_airplane_mode",
        "toggle_cellular_data",
        "restart_device",
        "reseat_sim_card",
        "reset_network_settings",
    ],
    "CHARGING": [
        "clean_charging_port",
        "test_alternative_charger",
        "restart_device",
        "force_restart",
    ],
    "POWER_BATTERY": [
        "restart_device",
        "check_storage_space",
        "update_os",
        "force_restart",
    ],
    "AUDIO": [
        "check_silent_mode",
        "adjust_volume_settings",
        "restart_device",
        "toggle_bluetooth",
    ],
    "DISPLAY": [
        "restart_device",
        "force_restart",
        "update_os",
    ],
    "SOFTWARE_APP": [
        "clear_cache",
        "restart_device",
        "update_app",
        "reinstall_app",
    ],
    "CRASH_FREEZE": [
        "force_restart",
        "check_storage_space",
        "update_os",
        "reinstall_app",
    ],
    "PERFORMANCE": [
        "restart_device",
        "check_storage_space",
        "update_os",
        "clear_cache",
    ],
    "ACCOUNT_ACCESS": [
        "check_spam_folder",
        "verify_apple_id",
        "update_os",
    ],
    "SYSTEM_UPDATE": [
        "check_storage_space",
        "restart_device",
        "forget_and_reconnect_wifi",
        "update_os",
    ],
    "STORAGE": [
        "check_storage_space",
        "clear_cache",
        "restart_device",
    ],
    "NOTIFICATION_ALERTS": [
        "check_silent_mode",
        "restart_device",
        "update_app",
    ],
    "GENERAL_DEVICE_FUNCTIONALITY": [
        "restart_device",
        "force_restart",
        "update_os",
    ],
}

DEFAULT_FALLBACK_SEQUENCE: List[str] = [
    "restart_device",
    "force_restart",
    "update_os",
]


class ActionCatalog:
    """
    Catalog of canonical troubleshooting actions with semantic alias recognition,
    progressive sequencing, and repeat-prevention verification.
    """

    @classmethod
    def identify_action(cls, text: str) -> Optional[str]:
        """
        Identify canonical action name from natural language text using regex and aliases.
        Returns the canonical action name if recognized, or None.
        """
        text_lower = text.lower()

        # Priority 1: Check compound or specific actions first
        specific_order = [
            "force_restart",
            "reset_network_settings",
            "forget_and_reconnect_wifi",
            "re_pair_bluetooth",
            "clean_charging_port",
            "test_alternative_charger",
            "check_airplane_mode",
            "reseat_sim_card",
            "check_silent_mode",
            "check_spam_folder",
            "verify_apple_id",
            "factory_reset",
            "clear_cache",
            "reinstall_app",
            "update_app",
            "update_os",
            "check_storage_space",
            "toggle_cellular_data",
            "toggle_bluetooth",
            "adjust_volume_settings",
            "restart_device",
        ]

        for canonical_name in specific_order:
            patterns = SEMANTIC_ACTION_ALIASES.get(canonical_name, [])
            for pat in patterns:
                if re.search(pat, text_lower, re.IGNORECASE):
                    return canonical_name

        return None

    @classmethod
    def is_equivalent(cls, action_a: str, action_b: str) -> bool:
        """
        Check if two action names or freeform action descriptions are semantically equivalent.
        E.g. "reboot" == "restart_device" == "power cycle" == "turned off and back on".
        """
        canonical_a = cls.get_canonical_name(action_a)
        canonical_b = cls.get_canonical_name(action_b)

        if canonical_a and canonical_b:
            return canonical_a == canonical_b

        # Fallback to direct string equality if neither is in catalog
        return action_a.strip().lower() == action_b.strip().lower()

    @classmethod
    def get_canonical_name(cls, action_text: str) -> Optional[str]:
        """Resolve a canonical name directly or by alias matching."""
        cleaned = action_text.strip().lower().replace(" ", "_")
        if cleaned in ACTION_METADATA:
            return cleaned

        identified = cls.identify_action(action_text)
        if identified:
            return identified

        return None

    @classmethod
    def get_action_metadata(cls, canonical_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata dict for a canonical action."""
        return ACTION_METADATA.get(canonical_name)

    @classmethod
    def is_destructive(cls, canonical_name: str) -> bool:
        """Check if an action is destructive (e.g. data loss risk)."""
        meta = ACTION_METADATA.get(canonical_name)
        if meta:
            return meta.get("is_destructive", False)
        return False

    @classmethod
    def get_progressive_sequence(cls, problem_family: Optional[str]) -> List[str]:
        """Get canonical progressive troubleshooting sequence for a problem family."""
        if not problem_family:
            return list(DEFAULT_FALLBACK_SEQUENCE)

        normalized = problem_family.upper().strip()
        if normalized in FAMILY_ACTION_SEQUENCES:
            return list(FAMILY_ACTION_SEQUENCES[normalized])

        return list(DEFAULT_FALLBACK_SEQUENCE)

    @classmethod
    def get_next_untried_action(
        cls,
        problem_family: Optional[str],
        attempted_canonical_names: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Return metadata of the next progressive action in sequence that has NOT been attempted.
        Returns None if all progressive actions have been exhausted.
        """
        sequence = cls.get_progressive_sequence(problem_family)
        for act_name in sequence:
            if act_name not in attempted_canonical_names:
                # Do not recommend destructive actions without explicit escalation gate
                meta = cls.get_action_metadata(act_name)
                if meta and not meta.get("is_destructive", False):
                    return meta
        return None
