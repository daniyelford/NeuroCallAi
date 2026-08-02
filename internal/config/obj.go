package config

import "sync"

type Config struct {
	App AppConfig `yaml:"app"`

	Server ServerConfig `yaml:"server"`

	Logging LoggingConfig `yaml:"logging"`
}

type AppConfig struct {
	Name        string `yaml:"name"`
	Environment string `yaml:"environment"`
}

type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
}

type LoggingConfig struct {
	Level string `yaml:"level"`
}
type Memory struct {
	mu     sync.RWMutex
	values map[string]string
}
