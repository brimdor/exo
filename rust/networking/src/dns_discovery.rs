//! DNS-based peer discovery for Kubernetes environments.
//!
//! This module provides an alternative to mDNS discovery that works in Kubernetes
//! by resolving a headless service DNS name to get peer IP addresses.

use futures_timer::Delay;
use libp2p::{Multiaddr, PeerId};
use std::collections::HashSet;
use std::env;
use std::net::{IpAddr, ToSocketAddrs};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Default port for libp2p connections
const DEFAULT_LIBP2P_PORT: u16 = 0; // Will be discovered from listening address

/// Configuration for DNS-based discovery
#[derive(Debug, Clone)]
pub struct DnsDiscoveryConfig {
    /// The DNS hostname to resolve (e.g., "exo-headless.exo.svc.cluster.local")
    pub hostname: String,
    /// Port to connect to on discovered peers
    pub port: u16,
    /// How often to poll DNS for updates
    pub poll_interval: Duration,
}

impl DnsDiscoveryConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Option<Self> {
        let mode = env::var("EXO_DISCOVERY_MODE").unwrap_or_else(|_| "mdns".to_string());
        
        if mode.to_lowercase() != "dns" {
            return None;
        }

        let hostname = env::var("EXO_DNS_HOSTNAME").ok()?;
        
        let port = env::var("EXO_DNS_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0); // 0 means use the same port as our listener
        
        let poll_interval = env::var("EXO_DNS_POLL_INTERVAL")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(5));

        Some(Self {
            hostname,
            port,
            poll_interval,
        })
    }
}

/// DNS discovery state for tracking known peers
#[derive(Debug, Default)]
pub struct DnsDiscoveryState {
    /// Known peer IPs from DNS resolution
    pub known_ips: HashSet<IpAddr>,
    /// Retry delay timer
    retry_delay: Option<Delay>,
    /// Configuration
    config: Option<DnsDiscoveryConfig>,
    /// Our own listening port (to use for dialing peers)
    pub listening_port: Option<u16>,
}

impl DnsDiscoveryState {
    /// Create a new DNS discovery state
    pub fn new() -> Self {
        Self {
            known_ips: HashSet::new(),
            retry_delay: None,
            config: DnsDiscoveryConfig::from_env(),
            listening_port: None,
        }
    }

    /// Check if DNS discovery is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    /// Get the poll interval
    pub fn poll_interval(&self) -> Duration {
        self.config
            .as_ref()
            .map(|c| c.poll_interval)
            .unwrap_or(Duration::from_secs(5))
    }

    /// Initialize the retry delay
    pub fn init_delay(&mut self) {
        if self.config.is_some() && self.retry_delay.is_none() {
            self.retry_delay = Some(Delay::new(Duration::from_secs(1))); // Initial quick poll
        }
    }

    /// Reset the retry delay
    pub fn reset_delay(&mut self) {
        let interval = self.poll_interval();
        if let Some(ref mut delay) = self.retry_delay {
            delay.reset(interval);
        }
    }

    /// Get the delay for polling
    pub fn take_delay(&mut self) -> Option<&mut Delay> {
        self.retry_delay.as_mut()
    }

    /// Resolve DNS hostname and return new peer addresses
    pub fn resolve_peers(&mut self) -> Vec<(IpAddr, u16)> {
        let Some(ref config) = self.config else {
            return vec![];
        };

        let port = if config.port == 0 {
            self.listening_port.unwrap_or(52414) // Default libp2p port if not set
        } else {
            config.port
        };

        // Resolve the hostname
        let socket_addr = format!("{}:{}", config.hostname, port);
        
        let resolved: Vec<(IpAddr, u16)> = match socket_addr.to_socket_addrs() {
            Ok(addrs) => {
                let new_addrs: Vec<_> = addrs
                    .filter_map(|addr| {
                        let ip = addr.ip();
                        if !self.known_ips.contains(&ip) {
                            self.known_ips.insert(ip);
                            Some((ip, port))
                        } else {
                            None
                        }
                    })
                    .collect();
                
                if !new_addrs.is_empty() {
                    info!(
                        "DNS discovery: found {} new peer(s) at {}",
                        new_addrs.len(),
                        config.hostname
                    );
                }
                
                new_addrs
            }
            Err(e) => {
                warn!("DNS discovery: failed to resolve {}: {}", config.hostname, e);
                vec![]
            }
        };

        resolved
    }

    /// Create a multiaddress for a given IP and port
    pub fn ip_to_multiaddr(ip: IpAddr, port: u16) -> Multiaddr {
        match ip {
            IpAddr::V4(ip4) => format!("/ip4/{}/tcp/{}", ip4, port).parse().unwrap(),
            IpAddr::V6(ip6) => format!("/ip6/{}/tcp/{}", ip6, port).parse().unwrap(),
        }
    }

    /// Get all known peer addresses for retry dialing
    pub fn get_all_peer_addrs(&self) -> Vec<Multiaddr> {
        let Some(ref config) = self.config else {
            return vec![];
        };

        let port = if config.port == 0 {
            self.listening_port.unwrap_or(52414)
        } else {
            config.port
        };

        self.known_ips
            .iter()
            .map(|ip| Self::ip_to_multiaddr(*ip, port))
            .collect()
    }
}

/// Check if DNS discovery mode is enabled
pub fn is_dns_discovery_enabled() -> bool {
    let mode = env::var("EXO_DISCOVERY_MODE").unwrap_or_else(|_| "mdns".to_string());
    mode.to_lowercase() == "dns"
}
