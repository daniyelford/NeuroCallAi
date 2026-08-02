package plugin

import "errors"

var (
	ErrNilPlugin = errors.New("plugin is nil")

	ErrEmptyPluginName = errors.New(
		"plugin name cannot be empty",
	)

	ErrDuplicatePlugin = errors.New(
		"plugin already registered",
	)

	ErrPluginNotFound = errors.New(
		"plugin not found",
	)

	ErrDependencyNotFound = errors.New(
		"plugin dependency not found",
	)

	ErrDependencyCycle = errors.New(
		"plugin dependency cycle detected",
	)
)
