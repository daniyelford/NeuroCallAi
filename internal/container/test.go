package container

import "testing"

func TestRegisterAndResolve(t *testing.T) {

	c := New()

	value := "hello"

	if err := c.Register("test", value); err != nil {
		t.Fatal(err)
	}

	got, err := c.Resolve("test")

	if err != nil {
		t.Fatal(err)
	}

	if got != value {
		t.Fatalf(
			"expected %v, got %v",
			value,
			got,
		)
	}
}

func TestDuplicateRegistration(t *testing.T) {

	c := New()

	if err := c.Register("test", "one"); err != nil {
		t.Fatal(err)
	}

	err := c.Register("test", "two")

	if err != ErrAlreadyRegistered {
		t.Fatalf(
			"expected ErrAlreadyRegistered, got %v",
			err,
		)
	}
}

func TestMissingDependency(t *testing.T) {

	c := New()

	_, err := c.Resolve("missing")

	if err != ErrNotFound {
		t.Fatalf(
			"expected ErrNotFound, got %v",
			err,
		)
	}
}
